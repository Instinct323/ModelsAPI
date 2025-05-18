import supervision as sv


def detection_labels(detections: sv.Detections):
    if detections.class_id is not None:
        return [
            f"{phrase} {score:.2f}"
            for phrase, score
            in zip(detections.metadata[detections.class_id], detections.confidence)
        ]
    else:
        return [f"{score:.2f}" for score in detections.confidence]
