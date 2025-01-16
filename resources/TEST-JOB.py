import os

host = os.getenv("HOSTNAME", "localhost")


from resources.train_ddp import trainer, ft_model_folder


if __name__ == "__main__":
    trainer.train()
    trainer.save_model(ft_model_folder)
