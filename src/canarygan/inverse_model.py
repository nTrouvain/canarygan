import numpy as np

from torch.utils import data
from rich.progress import track

from .generate import generate, generate_and_decode
from .canarygan.params import SliceLengths 


def is_garbage(label):
    prefixes = ["EARLY", "OT", "WN"]
    return any([p in label for p in prefixes])


def remap_garbage_classes(class_to_idx):

    x_idxs = [i for k, i in class_to_idx.items() if is_garbage(k)]
    mask = np.ones(len(class_to_idx))
    mask[x_idxs] = 0
    # shift class index to account for removed classes
    shift = np.cumsum(~mask.astype(bool)) 
    idxs = np.array(sorted(class_to_idx.values()))
    idx_remapping = dict(zip(idxs, idxs - shift))
    # remap classes to indexes and add a X class (last index)
    class_to_idx = {
        k: idx_remapping[i] 
        for k, i in class_to_idx.items()
        if not is_garbage(k)
    }
    class_to_idx["X"] = max(class_to_idx.values()) + 1 
    return class_to_idx


def filter_classes(p_r, class_to_idx):
    """Aggregate fake classes into one X class."""
    x_idxs = [i for k, i in class_to_idx.items() if is_garbage(k)]
    mask = np.ones(p_r.shape[-1])
    mask[x_idxs] = 0
    real_p_r = p_r[:, :, mask.astype(bool)]
    x_p_r = p_r[:, :, ~mask.astype(bool)]
    # Aggregation is performed using max
    x_p_r = np.max(x_p_r, axis=-1, keepdims=True)
    # Append aggregated garbage class (X)
    p_r = np.dstack([real_p_r, x_p_r])
    return p_r


def init_model(
    p_dim=17,
    m_dim=3,
    low=-0.001, 
    high=0.001,
):
    """
    W in R^(n_p x n_m)
    W ~ U[-0.001, 0.001]
    """
    return np.random.uniform(low=low, high=high, size=(p_dim, m_dim))


def normalize_p(p_r):
    """
    P_Ri = 1 if max(Y_Ri) > p95_Ri else max(Y_Ri) / p95_Ri
    """
    p_r = np.piecewise(
        p_r,
        [p_r < 0.01, (p_r >= 0.01) & (p_r <= 1), p_r > 1],
        [0.0, lambda x: x, 1.0],
    )
    return p_r


def rescale_p(p_r, p95):
    p_r = np.where(p_r > p95, 1.0, np.divide(p_r, p95))
    return p_r


def normalize_m(m_r):
    """
    M_ri = -1 if W 
    """
    m_r = np.piecewise(
        m_r, 
        [m_r < -1, (m_r <= 1) & (m_r >= -1), m_r > 1],
        [-1.0, lambda x: x, 1.0],
    )
    return m_r


def select_max(a_rs):
    """From Silvia's code.
    'Select the peak of the most active output' actually means
    keeping activity only at the timestep where peak activity
    (for any classes) happened.
    """
    def select(a_r):
        t_max, _ = np.unravel_index(np.argmax(a_r), a_r.shape)
        return a_r[t_max]

    return np.vstack([select(a_r) for a_r in a_rs])


def hebb_update(w, m_r, p_r, eta=0.01):
    """ 
    Hebbian update of W.
    dw = eta * P_rT . M_r
    """
    m_r = np.atleast_2d(m_r)
    p_r = np.atleast_2d(p_r)
    w += eta * np.dot(p_r.T,  m_r)
    return w


def hebb3f_update(w, m_r, p_r, xi, eta=0.01, goal=1.0, kind="hard"):
    """
    Hebbian update with stop criterion.
    dw = eta * (P_rT * xi). M_r
    with xi_i = 1 if Pri_t-1 < 1 else 0 (hard stop)
    or   xi_i = (1 - Pri_t-1)           (soft stop)
    or   xi_i = 1                       (none)
    """
    m_r = np.atleast_2d(m_r)
    p_r = np.atleast_2d(p_r)
    w += eta * np.dot(p_r.T * xi.T, m_r)

    return w


def inverse(w, p_r):
    p_r = np.atleast_2d(p_r)
    m_r = normalize_m(np.dot(w.T, p_r.T))
    return m_r.T


def create_learning_set(
    motor_vects,
    decoder,
    generator,
    activation,
    device,
    ):
    dataloader = data.DataLoader(
    motor_vects, batch_size=1024, shuffle=False, num_workers=12,
    )

    # Create the learning dataset: generate sounds and label them
    _, a_rs, m_rs = generate_and_decode(
        decoder,
        generator,
        dataloader,
        device=device,
        save_gen=True
    )
    max_time = round(0.5 * a_rs.shape[1]) # only 500ms

    a_rs = filter_classes(a_rs, decoder.class_to_idx) # Use X class for all bad productions
    a_rs = a_rs[:, :max_time, :-1] # Select first 0.5s and remove X ?

    if activation == "max":
        p_rs = select_max(a_rs)
    elif activation == "mean":
        p_rs = np.mean(a_rs, axis=1)
    else:
        raise NotImplementedError(f"activation: {activation}")

    p_rs = normalize_p(p_rs)

    p95 = np.percentile(p_rs, 95, axis=0) # 95-percentile, class-wise
    
    class_to_idx = remap_garbage_classes(decoder.class_to_idx)
    
    return p_rs, m_rs, p95, class_to_idx


def train_inverse_random(
    p_rs,
    m_rs,
    p95,
    w_dist,
    eta,
    max_steps,
    eval_every_n_steps,
    ):
    p_dim = p_rs.shape[1]
    m_dim = m_rs.shape[1]
    w = init_model(p_dim=p_dim, m_dim=m_dim, **w_dist)

    p_r_base = np.eye(p_dim)  # Used to probe inverse model for each perceptual class
    m_base_register = np.zeros((p_dim, max_steps // eval_every_n_steps, m_dim))
    w_history = np.zeros((max_steps // eval_every_n_steps,) + w.shape)
    
    for i in track(range(max_steps), "Training inverse"):
        p_r, m_r = p_rs[i], m_rs[i]
        p_r = rescale_p(p_r, p95)
        
        w = hebb_update(w, m_r, p_r, eta=eta)

        if i % eval_every_n_steps == 0:
            # Probe each perceptual category mapped by the inverse model
            m_r_base = inverse(w, p_r_base)
            m_base_register[:, i // eval_every_n_steps, :] = m_r_base
            w_history[i // eval_every_n_steps, :, :] = w
    
    return w, w_history, m_base_register


def update_achievement(xi, criterion, p_rs, p_gens, margin, tau, kind="hard"):
    
    reached = np.where(p_gens >= p_rs - margin, 1.0, 0.0)

    if kind == "plateau":
        # increase goal reach sum on plateau, reduce otherwise
        c_update = np.sign(p_gens - p_rs + margin)
        c_update = np.where(c_update == 0, 1, c_update)

        criterion += c_update
        criterion = np.where(criterion < 0.0, 0.0, criterion)

        xi = np.where(criterion >= tau, 0.0, xi)

        return xi, criterion, reached
    if kind == "hard":
        # Count successes before interruption
        d = np.where(p_gens >= p_rs - margin, 0.0, 1.0)
    elif kind == "soft":
        # Measure distance to goal in perceptual space
        d = (p_rs - margin) - p_gens
        d = np.where(d < 0.0, 0.0, d)  # in case of negative distance
    elif kind == "none":
        d = np.zeros_like(p_gens)
    else:
        raise NotImplementedError(f"kind: {kind}")

    criterion += 1/tau * d

    # Integrate goal achievement
    return xi + criterion, criterion, reached
    

def train_inverse_random_and_stop(
    p_rs,
    m_rs,
    actor_fn,
    p95,
    w_dist,
    eta,
    margin,
    kind,
    tau,
    max_steps,
    eval_every_n_steps,
    ):
    p_dim = p_rs.shape[1]
    m_dim = m_rs.shape[1]
    w = init_model(p_dim=p_dim, m_dim=m_dim, **w_dist)

    p_r_base = np.eye(p_dim)  # Used to probe inverse model for each perceptual class
    m_base_register = np.zeros((p_dim, max_steps // eval_every_n_steps, m_dim))
    w_history = np.zeros((max_steps // eval_every_n_steps,) + w.shape)
    x_gens_history = np.zeros((p_dim, max_steps // eval_every_n_steps, SliceLengths.SHORT.value))
    p_gens_history = np.zeros((p_dim, max_steps // eval_every_n_steps))

    # Nb of times a perceptual goal was reached
    reaches = np.zeros((p_dim, max_steps // eval_every_n_steps))
    # Criterion variable used to stop/slow down learning
    criterion = np.zeros((1, p_dim))
    # Goal achievement variables (all set to 1: incomplete)
    # xi_t = xi_t-1 * criterion
    xi = np.ones((1, p_dim))

    xi_history = np.zeros((p_dim, max_steps // eval_every_n_steps))
    criterion_history = np.zeros((p_dim, max_steps // eval_every_n_steps))

    for i in track(range(max_steps), "Training inverse"):
        p_r, m_r = p_rs[i], m_rs[i]
        p_r = rescale_p(p_r, p95)
        
        w = hebb3f_update(w, m_r, p_r, xi=xi, eta=eta)

        if i % eval_every_n_steps == 0:
            # Probe each perceptual category mapped by the inverse model
            m_r_base = inverse(w, p_r_base)
            x_gens, p_gens = actor_fn(m_rs=m_r_base[np.newaxis], p95=p95)
           
            p_gens = np.diagonal(p_gens, axis1=0, axis2=2)
            p_goal = np.ones_like(p_gens)
        
            xi, criterion, reached = update_achievement(
                xi, criterion, p_goal, p_gens, margin=margin, tau=tau, kind=kind
            )

            reaches[:, i // eval_every_n_steps] = reached
            criterion_history[:, i // eval_every_n_steps] = criterion
            m_base_register[:, i // eval_every_n_steps, :] = m_r_base
            x_gens_history[:, i // eval_every_n_steps, :] = x_gens
            p_gens_history[:, i // eval_every_n_steps] = p_gens
            w_history[i // eval_every_n_steps, :, :] = w
            xi_history[:, i // eval_every_n_steps] = xi
    
    return (
        w, 
        w_history,
        xi_history, 
        reaches, 
        criterion_history, 
        p_gens_history,
        x_gens_history,
        m_base_register
    )


def act_and_decode(
    decoder,
    generator,
    m_rs,
    p95,
    activation,
    device,
    ):    

    x_gens, a_rs, _ = generate_and_decode(
        decoder,
        generator,
        m_rs,
        device=device,
        save_gen=True,
    ) 
    max_time = round(0.5 * a_rs.shape[1]) # only 500ms
    a_rs = filter_classes(a_rs, decoder.class_to_idx) # Use X class for all bad productions
    a_rs = a_rs[:, :max_time, :-1]

    if activation == "max":
        p_rs = select_max(a_rs)
    elif activation == "mean":
        p_rs = np.mean(a_rs, axis=1)
    else:
        raise NotImplementedError(f"activation: {activation}")

    p_rs = rescale_p(normalize_p(p_rs), p95)
    
    # Go back to original shape (batches, steps, p_dim)
    p_dim = p_rs.shape[1]
    p_rs = p_rs.reshape(p_dim, -1, p_dim)

    return x_gens, p_rs


def train_inverse_model(
    decoder,
    generator,
    motor_vects,
    max_steps=3000,
    eta=0.01,
    w_dist={"low": -0.001, "high": 0.001},
    activation="max",
    device="cpu",
    eval_every_n_steps=15,
):

    dataloader = data.DataLoader(
    motor_vects, batch_size=1024, shuffle=False, num_workers=12,
    )

    # Create the learning dataset: generate sounds and label them
    _, a_rs, m_rs = generate_and_decode(
        decoder,
        generator,
        dataloader,
        device=device,
        save_gen=True
    )
    max_time = round(0.5 * a_rs.shape[1]) # only 500ms

    a_rs = filter_classes(a_rs, decoder.class_to_idx) # Use X class for all bad productions
    a_rs = a_rs[:, :max_time, :-1] # Select first 0.5s and remove X ?

    if activation == "max":
        p_rs = select_max(a_rs)
    elif activation == "mean":
        p_rs = np.mean(a_rs, axis=1)
    else:
        raise NotImplementedError(f"activation: {activation}")

    p_rs = normalize_p(p_rs)

    p95 = np.percentile(p_rs, 95, axis=0) # 95-percentile, class-wise
    
    class_to_idx = remap_garbage_classes(decoder.class_to_idx)

    p_dim = p_rs.shape[1]
    m_dim = m_rs.shape[1]
    w = init_model(p_dim=p_dim, m_dim=m_dim, **w_dist)

    p_r_base = np.eye(p_dim)  # Used to probe inverse model for each perceptual class
    m_base_register = np.zeros((p_dim, max_steps // eval_every_n_steps, m_dim))
    w_history = np.zeros((max_steps // eval_every_n_steps,) + w.shape)
    
    for i in track(range(max_steps), "Training inverse"):
        p_r, m_r = p_rs[i], m_rs[i]
        p_r = rescale_p(p_r, p95)
        
        w = hebb_update(w, m_r, p_r, eta)

        if i % eval_every_n_steps == 0:
            # Probe each perceptual category mapped by the inverse model
            m_r_base = inverse(w, p_r_base)
            m_base_register[:, i // eval_every_n_steps, :] = m_r_base
            w_history[i // eval_every_n_steps, :, :] = w
    
    x_gens, a_rs, _ = generate_and_decode(
        decoder,
        generator,
        m_base_register,
        device=device,
        save_gen=True,
    ) 

    a_rs = filter_classes(a_rs, decoder.class_to_idx) # Use X class for all bad productions
    a_rs = a_rs[:, :max_time, :-1]

    if activation == "max":
        p_rs = select_max(a_rs)
    elif activation == "mean":
        p_rs = np.mean(a_rs, axis=1)
    else:
        raise NotImplementedError(f"activation: {activation}")

    p_rs = rescale_p(normalize_p(p_rs), p95)
    
    # Go back to original shape (batches, steps, p_dim)
    p_rs = p_rs.reshape(p_dim, -1, p_dim)

    return w, w_history, m_base_register, x_gens, p_rs, class_to_idx, p95
