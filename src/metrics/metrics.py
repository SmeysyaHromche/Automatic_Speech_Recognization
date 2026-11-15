import jiwer

def compute_wer(all_targets:list[str], all_predictions:list[str]) -> float:
    all_predictions = [
        " ".join(text.upper().split())
        for text in all_predictions
    ]

    all_targets = [
        " ".join(text.upper().split())
        for text in all_targets
    ]

    wer = jiwer.wer(all_targets, all_predictions)

    return wer