from pathlib import Path
from matplotlib import pyplot as plt


def format_label(config: dict) -> str:
  opt = config["optimizer"]
  lr = config["learning_rate"]
  layers = config["layer"]
  return f"{opt} | lr={lr} | layers={layers}"


def visualisation(y_1, y_2, config1, config2, title):
  x_1 = list(range(1, len(y_1) + 1))
  x_2 = list(range(1, len(y_2) + 1))

  # Création du dossier de sauvegarde s'il n'existe pas encore
  Path("visu/compare").mkdir(parents=True, exist_ok=True)

  plt.figure(figsize=(10, 6))
  plt.plot(
      x_1,
      y_1,
      label=format_label(config1),
      color="red",
      linewidth=1.5,
      alpha=0.8,
  )
  plt.plot(
      x_2,
      y_2,
      label=format_label(config2),
      color="blue",
      linewidth=1.5,
      alpha=0.8,
  )

  plt.title(f"Comparison - {title}")
  plt.xlabel("Epochs")
  plt.ylabel(title)
  plt.legend()
  plt.grid(True)

  plt.savefig(f"visu/compare/{title}.png")
  plt.show()
  plt.close()


def compare(payload1, payload2):
  metrics1 = payload1["history"]
  metrics2 = payload2["history"]
  config1 = payload1["config"]
  config2 = payload2["config"]  # Bug corrigé ici

  visualisation(
      metrics1["val_acc"], metrics2["val_acc"], config1, config2, "val_acc"
  )
  visualisation(
      metrics1["val_loss"], metrics2["val_loss"], config1, config2, "val_loss"
  )
