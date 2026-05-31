def weighted_score(xlmr_score, stylometry_score):
    final_score = (0.80 * xlmr_score) + (0.20 * stylometry_score)
    return round(final_score, 4)
