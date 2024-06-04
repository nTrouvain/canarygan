from canarygan.gan import train


def test_training(tmp_dummy_data, tmp_save_dir):
    train(
        save_dir=tmp_save_dir,
        data_dir=tmp_dummy_data,
        max_epochs=1,
        batch_size=16,
        devices=1,
        num_nodes=1,
        num_workers=1,
        log_every_n_steps=1,
        save_every_n_epochs=1,
        save_topk=1,
        resume=False,
        seed=0,
        version="infer",
        dry_run=False,
    )

    assert (tmp_save_dir / "checkpoints").exists()
    assert (tmp_save_dir / "logs").exists()
