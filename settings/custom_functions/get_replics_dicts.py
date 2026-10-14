def get_replics_dicts():

    replics_histone = {'spKAS-seq': ['spKAS-seq.rep1', 'spKAS-seq.rep2'],

                      }



    replics_speckle = {'LaminAC_TSA-Seq' : [
        'GSE81553_LaminAC_TSA-Seq_Condition2',
        'GSE81553_LaminAC_TSA-Seq_Conditoin2_Replicate'],

                       'LaminB_TSA-Seq' : [
                           'GSE81553_LaminB_TSA-Seq_Condition2', 
                           'GSE81553_LaminB_TSA-Seq_Conditoin2_Replicate'],

                       'Pol2_TSA-Seq' : ['GSE81553_Pol2_TSA-Seq_Condition1','GSE81553_Pol2_TSA-Seq_Condition1_Replicate'],

                       'pSC35_TSA-Seq' : [ 'GSE81553_pSC35_TSA-Seq_Condition1','GSE81553_pSC35_TSA-Seq_Condition1_Replicate'],

                       'SON_TSA-Seq' : ['GSE81553_SON_TSA-Seq_Ab2_Condition2','GSE81553_SON_TSA-Seq_Condition1','GSE81553_SON_TSA-Seq_Condition1_Replicate',
                                        'GSE81553_SON_TSA-Seq_Condition2','GSE81553_SON_TSA-Seq_Condition2.Replicate','GSE81553_SON_TSA-Seq_Condition3'],

                       'Speckle_distance' :['GSE81553_Speckle_distance_Condition1','GSE81553_Speckle_distance_Condition2','GSE81553_Speckle_distance_Condition3']
                       }

    replics_u2af2 = {'SF3B1_CUTnRUN' : ['GSM8339584_K562_SF3B1_Minus_RNAseA_CUTnRUN_rep1.hg19', 
                                        'GSM8339585_K562_SF3B1_Minus_RNAseA_CUTnRUN_rep2.hg19'],

                     'U2AF1_CUTnRUN': [ 'GSM8339588_K562_U2AF1_Minus_RNAseA_CUTnRUN_rep1.hg19',
                                         'GSM8339589_K562_U2AF1_Minus_RNAseA_CUTnRUN_rep2.hg19'],

                     'U2AF2_CUTnRUN': [ 'GSM8339592_K562_U2AF2_Minus_RNAseA_CUTnRUN_rep1.hg19',
                                        'GSM8339593_K562_U2AF2_Minus_RNAseA_CUTnRUN_rep2.hg19',
                                         'GSM8339594_K562_U2AF2_Minus_RNAseA_CUTnRUN_rep3.hg19',
                                         'GSM8339598_K562_U2AF2_CUTnRUN_1.hg19',
                                         'GSM8339599_K562_U2AF2_CUTnRUN_2.hg19',
                                         'GSM8339600_K562_U2AF2_CUTnRUN_3.hg19',
                                         'GSM8339601_K562_U2AF2_CUTnRUN_4.hg19',
                                         'GSM8339602_K562_U2AF2_CUTnRUN_5.hg19',
                                        'GSM8339603_K562_treated_with_DMSO_U2AF2_CUTnRUN_1.hg19',
                                         'GSM8339604_K562_treated_with_DMSO_U2AF2_CUTnRUN_2.hg19',
                                         'GSM8339611_K562_treated_with_DMSO_U2AF2_CUTnRUN_3.hg19',
                                         'GSM8339612_K562_treated_with_DMSO_U2AF2_CUTnRUN_4.hg19'],

                     'H3K36me3_CUTnRUN' : ['GSM8339615_K562_H3K36me3_CUTnRUN_rep1.hg19',
                                           'GSM8339616_K562_H3K36me3_CUTnRUN_rep2.hg19', 
                                           'GSM8343601_K562_H3K36me3_CUTnTAG_1.hg19',
                                             'GSM8343602_K562_H3K36me3_CUTnTAG_2.hg19',
                                             'GSM8343603_K562_H3K36me3_CUTnTAG_3.hg19',
                                             'GSM8343604_K562_H3K36me3_CUTnTAG_4.hg19',
                                             'GSM8343605_K562_H3K36me3_CUTnTAG_5.hg19',
                                             'GSM8343606_K562_treated_with_DMSO_H3K36me3_CUTnTAG_rep1.hg19',
                                             'GSM8343607_K562_treated_with_DMSO_H3K36me3_CUTnTAG_rep2.hg19'],

                     'RNAPIIS2P_CUTnRUN': ['GSM8339617_K562_RNAPIIS2P_CUTnRUN_rep1.hg19',
                                           'GSM8339618_K562_RNAPIIS2P_CUTnRUN_rep2.hg19',
                                           'GSM8343588_K562_RNAPIIS2P_CUTnTAG.hg19',
                                           'GSM8343593_K562_treated_with_DMSO_RNAPIIS2P_CUTnTAG_rep1.hg19',
                                           'GSM8343594_K562_treated_with_DMSO_RNAPIIS2P_CUTnTAG_rep2.hg19', 
                                           'GSM8917357_K562_RNAPIIS2P_Minus_RNAseA_CUTnTag_1.hg19',
                                           'GSM8917358_K562_RNAPIIS2P_Minus_RNAseA_CUTnTag_2.hg19',
                                           'GSM8917359_K562_RNAPIIS2P_Minus_RNAseA_CUTnTag_3.hg19',
                                           'GSM8917360_K562_RNAPIIS2P_plus_RNAseA_CUTnTag_1.hg19',
                                           'GSM8917361_K562_RNAPIIS2P_plus_RNAseA_CUTnTag_2.hg19',
                                           'GSM8917362_K562_RNAPIIS2P_plus_RNAseA_CUTnTag_3.hg19'],

                     'SRSF1_CUTnRUN' : ['GSM8917353_K562_SRSF1_CUTnRUN_1.hg19',
                                        'GSM8917354_K562_SRSF1_CUTnRUN_2.hg19'],

                     'SRSF3_CUTnRUN' : [ 'GSM8917355_K562_SRSF3_CUTnRUN_1.hg19',
                                        'GSM8917356_K562_SRSF3_CUTnRUN_2.hg19'],

                     'RNAPIIS5P_CUTnTAG' : ['GSM8343589_K562_treated_with_DMSO_RNAPIIS5P_CUTnTAG.hg19']}

    replics_stranded = {'RNAseqPolyA' : ['RNAseqPolyA_1', 'RNAseqPolyA_2'],
                        'RNAseqPolyAminus' : ['RNAseqPolyAminus_1', 'RNAseqPolyAminus_2'],
                        'RNAseqPolyAnucleolus' : ['RNAseqPolyAnucleolus_1', 'RNAseqPolyAnucleolus_2'],
                        'RNAseqChromatin' : ['RNAseqChromatin_1', 'RNAseqChromatin_2'],
                        'GRO' : ['GRO_1', 'GRO_2'],
                        'NET_Meyer_input' : ['NET_Meyer_input1','NET_Meyer_input2']
                    }
    replics_dicts = {
        "histone": replics_histone,
        "speckle": replics_speckle,
        "u2af2": replics_u2af2,
        "stranded": replics_stranded,
    }
    
    return replics_dicts