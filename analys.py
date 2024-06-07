import os
import json


def compare_points(ground_truth, predictions, tolerance):
    total_objects = len(ground_truth)
    correct_predictions = 0

    for obj_gt in ground_truth:
        for obj_pred in predictions:
            if obj_gt["classTitle"] == obj_pred["classTitle"]:
                matched_points = 0
                for point_gt in obj_gt["points"]["exterior"]:
                    for point_pred in obj_pred["points"]["exterior"]:
                        if all(abs(p_gt - p_pred) <= tolerance for p_gt, p_pred in zip(point_gt, point_pred)):
                            matched_points += 1
                            break

                if matched_points == len(obj_gt["points"]["exterior"]):
                    correct_predictions += 1
                    break

    accuracy = correct_predictions / total_objects if total_objects > 0 else 0
    return accuracy


def compare_points_by_class_title(ground_truth, predictions):
    class_title_correct = {}
    total_class_titles = {}

    for obj_gt in ground_truth:
        class_title = obj_gt["classTitle"]
        total_class_titles[class_title] = total_class_titles.get(class_title, 0) + 1

        for obj_pred in predictions:
            if obj_gt["classTitle"] == obj_pred["classTitle"]:
                matched_points = 0
                for point_gt in obj_gt["points"]["exterior"]:
                    for point_pred in obj_pred["points"]["exterior"]:
                        if all(abs(p_gt - p_pred) <= tolerance for p_gt, p_pred in zip(point_gt, point_pred)):
                            matched_points += 1
                            break

                if matched_points == len(obj_gt["points"]["exterior"]):
                    class_title_correct[class_title] = class_title_correct.get(class_title, 0) + 1
                    break

    class_title_accuracy = {k: v / total_class_titles[k] if k in total_class_titles and total_class_titles[k] > 0 else 0
                            for k, v in class_title_correct.items()}
    return class_title_accuracy

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


ground_truth_folder = r'C:\Users\delat\PycharmProjects\objectDetection\val\ann'
predictions_folder = r'C:\Users\delat\PycharmProjects\objectDetection\val\predict'

tolerance = 20

ground_truth_file_paths = [os.path.join(ground_truth_folder, file) for file in os.listdir(ground_truth_folder) if
                           file.endswith('.json')]
predictions_file_paths = [os.path.join(predictions_folder, file) for file in os.listdir(predictions_folder) if
                          file.endswith('.json')]

total_accuracy = 0

for ground_truth_path, prediction_path in zip(ground_truth_file_paths, predictions_file_paths):
    ground_truth = read_json_file(ground_truth_path)['objects']
    predictions = read_json_file(prediction_path)

    accuracy = compare_points(ground_truth, predictions, tolerance)
    total_accuracy += accuracy

    print(f"Точность для файлов {ground_truth_path} и {prediction_path}: {accuracy}")

if len(ground_truth_file_paths) > 0:
    final_accuracy = total_accuracy / len(ground_truth_file_paths)
else:
    final_accuracy = 0

print(f"Общий коэффициент правильности: {final_accuracy}")

class_title_accuracy = compare_points_by_class_title(ground_truth, predictions)

for class_title, accuracy in class_title_accuracy.items():
    print(f"Точность для {class_title}: {accuracy}")