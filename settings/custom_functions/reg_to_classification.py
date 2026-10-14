def devide_into_intervals(value, mean_or_med_exp):
    if value == 0:
        return 0
    elif value < mean_or_med_exp:
        return 1
    else:  # value >= mean_exp
        return 2
    
def devide_into_binary(value, threshold):
    return 1 if value > threshold else 0


# Tertiles of the splicing fraction pooled over all eight introns, measured 2026-08-27 on the
# 298,451 defined values of `tables_for_training/intron_splicing/*.alldata.tab`: the 33.3rd
# percentile is 0.2876 and the 66.7th is 0.6916.
#
# They are CONSTANTS, not a rule to recompute. Fixed cut points mean a class means the same thing
# in every intron, which is what makes the introns comparable. Per-intron tertiles would balance
# each one and destroy exactly that. The per-intron balance therefore still varies (20/27/53 for
# B3-6 against 46/28/25 for E1), and that is real biology, not a defect to tune away.
#
# The previous 0.6 / 0.8 gave 59 / 15 / 26 pooled and skewed in OPPOSITE directions across
# introns, so a weighted-AUC comparison between introns compared different problems. These give
# 33 / 33 / 33 pooled.
SPLICING_LOW, SPLICING_HIGH = 0.288, 0.692


def devide_into_intervals_intron(value):
    if value < SPLICING_LOW:
        return 0
    elif value <= SPLICING_HIGH:
        return 1
    else:
        return 2
    