from pathlib import Path

from ultralytics import YOLO


def main():
    # Базовая COCO-pretrained модель
    model_name = "11m"
    model = YOLO(f"yolo{model_name}.pt")

    project_dir = Path("models/real_dataset_baselines")
    run_name = f"yolo{model_name}_640_e50_real_dataset"

    model.train(
        task="detect",
        data="Datasets/subset_001_yolo/data.yaml",
        imgsz=640,
        epochs=50,
        batch=8,
        device=0,
        workers=4,
        seed=42,

        # обучение от предобученной YOLO
        pretrained=True,

        # оптимизация
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        cos_lr=True,

        # полезно для детекции мелких объектов
        close_mosaic=10,
        rect=False,

        # экономия памяти / ускорение
        amp=True,
        cache=False,

        # валидация и сохранение
        val=True,
        plots=True,
        save=True,
        save_period=10,
        patience=30,

        project=str(project_dir),
        name=run_name,
        exist_ok=True,
    )

    # best_pt = project_dir / run_name / "weights" / "best.pt"
    # last_pt = project_dir / run_name / "weights" / "last.pt"
    #
    # print("\nTraining finished.")
    # print(f"Best weights: {best_pt}")
    # print(f"Last weights: {last_pt}")
    #
    # # Финальная проверка best.pt на val split
    # best_model = YOLO(str(best_pt))
    # best_model.val(
    #     data="VisDrone.yaml",
    #     imgsz=640,
    #     batch=8,
    #     device=0,
    #     workers=4,
    #     conf=0.001,
    #     iou=0.6,
    #     max_det=300,
    #     split="val",
    #     plots=True,
    # )
    #

if __name__ == "__main__":
    main()