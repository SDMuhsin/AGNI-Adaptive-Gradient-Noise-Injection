
import torch
import transformers
from models.bert.modeling_bert import BertForSequenceClassification
from models.roberta.modeling_roberta import RobertaForSequenceClassification
from models.distilbert.modeling_distilbert import DistilBertForSequenceClassification
from models.doublebert.modeling_doublebert import DoubleBertForSequenceClassification, DoubleBertForSequenceClassificationV2, DoubleBertForSequenceClassificationV3, DoubleBertForSequenceClassificationV4
from transformers.models.mobilebert.modeling_mobilebert import MobileBertForSequenceClassification
from transformers import AutoModelForSequenceClassification, AutoTokenizer
def create_model(config, model_args):
    model_name = model_args.model_name_or_path
    
    if model_name.startswith("double-bert-v4"):
    
        return DoubleBertForSequenceClassificationV4(config)
    elif model_name.startswith("double-bert-v3"):
    
        return DoubleBertForSequenceClassificationV3(config)

    elif model_name.startswith("double-bert-v2"):

        return DoubleBertForSequenceClassificationV2(
                config
                )
    elif model_name.startswith("double"):
        return DoubleBertForSequenceClassification(
                config

            )
    elif model_name.startswith("albert"):
        return AutoModelForSequenceClassification.from_pretrained(
                model_name,
                config=config,
                use_auth_token=None
                ) 

    elif model_name.startswith("bert") or model_name == "huawei-noah/TinyBERT_General_6L_768D":
        return BertForSequenceClassification.from_pretrained(
                model_name,
                from_tf=bool(".ckpt" in model_name),
                config=config,
                use_auth_token=None,
            )
    elif model_name.startswith("roberta"):
        return RobertaForSequenceClassification.from_pretrained(
                model_name,
                from_tf=bool(".ckpt" in model_name),
                config=config,
                use_auth_token=None,
            )
    elif model_name.startswith("distilbert"):
        return DistilBertForSequenceClassification.from_pretrained(
                model_name,
                from_tf=bool(".ckpt" in model_name),
                config=config,
                use_auth_token=None,
            )
    elif model_name.startswith("google/mobilebert-uncased"):
        return MobileBertForSequenceClassification.from_pretrained(
                model_name,
                from_tf=bool(".ckpt" in model_name),
                config=config,
                use_auth_token=False,
            )

    raise Exception(f"Model {model_name} unknown.")

import os
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
def create_model_load_save(config, model_args,model_path):


    
    model = None
    model_name = model_args.model_name_or_path
    if model_name.startswith("bert") or model_name == "huawei-noah/TinyBERT_General_6L_768D":

        if (os.path.exists(model_path)):
            return BertForSequenceClassification.from_pretrained(model_path)
        else:
            model = BertForSequenceClassification.from_pretrained(
                    model_name,
                    from_tf=bool(".ckpt" in model_name),
                    config=config,
                    cache_dir=model_args.cache_dir,
                    revision=model_args.model_revision,
                    use_auth_token=True if model_args.use_auth_token else None,
                )
    elif model_name.startswith("roberta"):
        if (os.path.exists(model_path)):
            return RobertaForSequenceClassification.from_pretrained(model_path)
        else:
            model = RobertaForSequenceClassification.from_pretrained(
                    model_name,
                    from_tf=bool(".ckpt" in model_name),
                    config=config,
                    cache_dir=model_args.cache_dir,
                    revision=model_args.model_revision,
                    use_auth_token=True if model_args.use_auth_token else None,
                )
    elif model_name.startswith("distilbert"):
        model = DistilBertForSequenceClassification.from_pretrained(
                model_name,
                from_tf=bool(".ckpt" in model_name),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )
    elif model_name.startswith("google/mobilebert-uncased"):
        model = MobileBertForSequenceClassification.from_pretrained(
                model_name,
                from_tf=bool(".ckpt" in model_name),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )
    if(model == None):
        raise Exception(f"Model {model_name} unknown.")
    
    model.save_pretrained(model_path)
    return model
