def get_groups_of_features():

    groups_of_features = {
        "accessibility": [
            "ATAC",
            "DNAse",
            "GSE81553_K562_HiC_100kb_Eigenvector"
        ],

        "histone_marks": [
            "H2AFZ",
            "H3K27ac",
            "H3K27me3",
            "H3K36me3",
            "H3K4me1",
            "H3K4me2",
            "H3K4me3",
            "H3K79me2",
            "H3K9ac",
            "H3K9me1",
            "H3K9me3",
            "H4K20me1",
            "H3K36me3_CUTnRUN"
        ],

        "pol2": [
            "ENCSR000BKR-POLR2AphosphoS5",
            "ENCSR000BMR-POLR2A",
            "ENCSR000DLY-POLR2A",
            "ENCSR000EGF-POLR2AphosphoS2",
            "ENCSR000EHF-POLR2AphosphoS2",
            "ENCSR000EHL-POLR2A",
            "ENCSR000FAJ-POLR2A",
            "ENCSR031TFS-POLR2A",
            "ENCSR388QZF-POLR2A",
            "GSE81553_K562_Smoothed_Pol2_ChIP-Seq",
            "RNAPIIS2P_CUTnRUN",
            "RNAPIIS5P_CUTnTAG",
            "Pol2_TSA-Seq"
        ],

        "transcriptomics": [
            "RNAseqPolyA",
            "RNAseqPolyAminus",
            "RNAseqPolyAnucleolus",
            "RNAseqChromatin",
            "GSE81553_K562_RNA-Seq_binned_20kb",
            "GSE81553_K562_GRO-Seq_binned_20kb",
            "PRO_Lis",
            "NET_Meyer_original",
            "NET_Meyer_input",   # note: control/input, treat separately in preprocessing if needed
            "GRO",
            "spKAS-seq",
            'icSHAPE',
             "K562.Cell"
        ],

        "nuclear_org": [
            "LaminAC_TSA-Seq",
            "LaminB_TSA-Seq",
            "pSC35_TSA-Seq",
            "SON_TSA-Seq",
            "Speckle_distance"

        ],

        'splicing': ["SF3B1_CUTnRUN",
            "U2AF1_CUTnRUN",
            'U2AF2_CUTnRUN',
            "SRSF1_CUTnRUN",
            "SRSF3_CUTnRUN"
                    ],

        'conservation': ['phyloP'],

        "annotations": [        
            "DistToTSS",
            "DistToTES",
            "RelPos",
            "DistToIntron5",
            "DistToIntron3",
            "RelIntronPos",
            "MANElength",
            "MANE_intron_length",
            'MANE_all',
            'MANE_exonic'
        ],

        "gc_content": [
            "GC"
        ],

        "chromhmm": [
            "ChromHMM"
        ],

        "mane_expression": [
            "MANE_all_exp.norm.log10"
        ],

        "global_norm": [
            "norm"
        ]
    }
    
    return groups_of_features