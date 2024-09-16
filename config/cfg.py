A_config = {
    #-----------------------------
    #中文候选项生成工具包config
    'custom_jieba' : True,
    'custom_letter_path' : '/home/zhangwentao/host/project/paper-v1.3/data/vocab',
    'letter_map_cn_path' : '/home/zhangwentao/host/project/paper-v1.3/data/vocab',
    'top_k_split'        : 4,
    #'field_data_path'    : '/home/zhangwentao/host/project/paper/data/vocab/sql_store.txt',
    #'key_word_path'      : '/home/zhangwentao/host/project/paper-v1.3/data/MedicalTwinTower/key_word/keyWord.txt'
    
    #-----------------------------
    #双塔模型 config
    'left_model_path'  : '/home/zhangwentao/host/project/paper-v1.3/module/twin_module/model/model/left_bert',
    'left_model_weight_path' : '/home/zhangwentao/host/project/paper-v1.3/module/twin_module/model/output/left_bert.pt',
    'left_hidden_size' : 768,
    'n_layers'         : 12,
    'attn_heads'       : 12,
    'seq_len'          : 128,

    'right_model_path' : '/home/zhangwentao/host/project/paper-v1.3/module/twin_module/model/model/right_bert/bge-large-zh-v1.5',

    'TwinTower_model_weight_path' : '/home/zhangwentao/host/project/paper-v1.3/module/twin_module/model/output/v3/weight-large/test_best.pt',
    'hidden_size'      : 1024,
    'mergeNet_layer'   : 1,
    'twin_batch_size'  : 4096,

    'device' : 'cuda:0',

    #----------------------------
    #LLM优化 config 
    'LLM_Model_Path' : '/home/zhangwentao/host/LLMs/Models/chatglm3-6b',
    'LLM_device'     : 'cuda:0',
    'local_infer'    : True,
    'batch_size'     : 32,
    'n_return'       : 100,

    # 去除双塔模型环节，直接用大模型推理
    'cot_infer'      : True
}