from pylabel import importer
dataset = importer.ImportVOC(path='datasets/BSTLD/pascal', path_to_images='../dataset_train_rgb/pascal')
dataset.export.ExportToYoloV5(output_path='datasets/BSTLD_yolo/train/labels', copy_images=True, use_splits=True, cat_id_index=0)