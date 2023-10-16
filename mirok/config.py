
class Config:
    def __init__(self,
            iterations=50,
            min_support=5,
            min_conf=0.85,
            max_res_op_num=5,
            stride=0.5,
            
            max_epochs=100, 
            epoch_step=100,
            activating_emb_epoch=101,
            early_stopping=3,
            ensemble_num=2,
            pretrained_path=None,
            dim_word_emb=100,
            dim_type_emb=30,
            dim_hidden=128,
            depth=2,
            valid_ratio=0.1,
            oversampling_count=500,
            oversampling_times=50,
            lr=0.001, 
            penalty_coef=0.2,
            weight_decay=1e-5,
            dropout=0.8,
            training_batch=64,
            predicting_batch=128,
            device="cuda"
        ):
        self.iterations = iterations
        self.min_support = min_support
        self.min_conf = min_conf
        self.max_res_op_num = max_res_op_num
        self.stride = stride
        self.max_epochs = max_epochs
        self.epoch_step = epoch_step
        self.activating_emb_epoch = activating_emb_epoch
        self.early_stopping = early_stopping
        self.ensemble_num = ensemble_num
        self.pretrained_path = pretrained_path
        self.dim_word_emb = dim_word_emb
        self.dim_type_emb = dim_type_emb
        self.dim_hidden = dim_hidden
        self.depth = depth
        self.valid_ratio = valid_ratio
        self.oversampling_count = oversampling_count
        self.oversampling_times = oversampling_times
        self.lr = lr
        self.penalty_coef = penalty_coef
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.training_batch = training_batch
        self.predicting_batch = predicting_batch
        self.device = device
        
    def __str__(self):
        return "=============================== CONFIG ==============================\n" + \
                "\n".join(f"{k}: {v}" for k, v in self.__dict__.items())
        

        