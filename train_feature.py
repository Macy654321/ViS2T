
from model.clipcap import *
from dataset.mydataloader import *
from torch.optim import AdamW
import math
import logging
import os
from datetime import datetime
from model.clipcap import *
def train():
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dataset = 'panda'
    test = 0
    load_model = 0
    frame = 30
    vocab_size = 50257
    log_dir = "logs"
    date_dir = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    save_dir = os.path.join(log_dir, date_dir)
    os.makedirs(save_dir, exist_ok=True)
    log_filename = "logs.log"
    log_path = os.path.join(save_dir, log_filename)
    # Translated comment.
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    # Translated comment.
    logging.info("begin training")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gpt2_model_path = r"E:\pycharmProject\clipcap\pretrained\gpt2"
    # Translated comment.
    model,tokenizer = define_feature_model(gpt2_model_path,device)
    model.to(device)

    if load_model:
        model.load_state_dict(torch.load("logs/25-09-12-12-34-36/model_last.pth"))
    if dataset == 'MSRVTT':
        max_len = 100
        batch_size = 16
        csv_file = 'train_data.csv'
        if test:
            csv_file = 'train_data_t.csv'
            batch_size = 2
        train_dataset = MsrvttData_feature('dataset\\msrvtt','vitb32_1fps', csv_file,
                                           caption_json="caption_all.json", tokenizer=tokenizer, max_len=max_len,frame=frame)
        test_dataset = MsrvttData_feature('dataset\\msrvtt','vitb32_1fps' , 'test_data.csv',
                                          caption_json="caption_all.json", tokenizer=tokenizer, max_len=max_len,frame=frame)
    elif dataset == 'panda':
        max_len = 75
        batch_size = 16
        csv_file = 'train_data.csv'
        feature_dir="lora_detail"
        if test:
            csv_file = 'train_data_t.csv'
            batch_size = 2
        train_dataset = Panda_feature('dataset\\panda',feature_dir, csv_file,
                                           caption_json="20250808.json", tokenizer=tokenizer,cap_type='short', max_len=max_len,frame=frame)
        test_dataset = Panda_feature('dataset\\panda',feature_dir , 'test_data.csv',
                                          caption_json="20250808.json", tokenizer=tokenizer,cap_type='short', max_len=max_len,frame=frame)
    else:#msvd
        max_len = 60
        batch_size = 16
        csv_file = 'train_data.csv'
        if test:
            csv_file = 'train_data_t.csv'
            batch_size = 2
        train_dataset = MsrvttData_feature('dataset\\msvd','vit_1fps', csv_file,
                                           caption_json="caption.json", tokenizer=tokenizer, max_len=max_len,frame=frame)
        test_dataset = MsrvttData_feature('dataset\\msvd','vit_1fps' , 'test_data.csv',
                                          caption_json="caption.json", tokenizer=tokenizer, max_len=max_len,frame=frame)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    if test:
        gene_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=4)
    else:
        gene_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    criterion = nn.CrossEntropyLoss( ignore_index=0,label_smoothing=0.01)
    # optimizer = AdamW([
    #     {"params": model.projection_layer.parameters(), "lr": base_lr}
    # ])
    optimizer = torch.optim.AdamW([
        {"params": model.projection_layer.parameters(), "lr": 2e-5, "weight_decay": 0.01},
        {"params": model.tqformer.parameters(), "lr": 2e-5, "weight_decay": 0.01},
        {"params": model.temsel.parameters(), "lr": 2e-5, "weight_decay": 0.01},
        {"params": model.fusion.parameters(), "lr": 2e-5, "weight_decay": 0.01},
        {"params": model.gpt2_model.parameters(), "lr": 1e-5, "weight_decay": 0.0}
    ])
    max_epoch = 40
    def cosine_decay(epoch):
        base_lr = 0.01
        k1=0.5-base_lr*0.5
        return k1 * (1 + math.cos(math.pi * epoch / max_epoch))+base_lr

    # Translated comment.
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=cosine_decay)
    simulate_batch =1
    min_loss = 1e10
    for epoch in range(max_epoch):
        model.train()
        optimizer.zero_grad()
        batch_cnt = 0
        test_loss = 0
        train_loss = 0
        ret_dict={}
        for i, batch in enumerate(train_loader):
            batch_cnt =i
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)

            ret = model(batch)
            labels = batch['target']
            logits = ret.logits[:, -(max_len+1):, :]
            loss = criterion(logits.reshape(-1, vocab_size), labels.reshape(-1))
            loss.backward()
            batch_loss = loss.item()
            train_loss += batch_loss
            if batch_cnt % 10 == 0:
                logging.info("batch_loss:%f",batch_loss)
                print("batch_loss:",batch_loss)
            if i % simulate_batch == 0:
                optimizer.step()
                optimizer.zero_grad()
        if batch_cnt % simulate_batch != 0:
            optimizer.step()
            optimizer.zero_grad()
        logging.info("train_loss:%f",train_loss/len(train_loader))
        print("train_loss:",train_loss/len(train_loader))
        scheduler.step()
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device)
                ret = model(batch)
                labels = batch['target']
                logits = ret.logits[:, -(max_len + 1):, :]
                loss = criterion(logits.reshape(-1, vocab_size), labels.reshape(-1))
                test_loss += loss.item()
                if test:
                    break
        logging.info("test_loss:%f",test_loss/len(test_loader))
        print("test_loss:",test_loss/len(test_loader))

        if test_loss<min_loss:
            min_loss=test_loss
            if not test:
                torch.save(model.state_dict(), os.path.join(save_dir,"model_best.pth"))
        else:
            if not test:
                torch.save(model.state_dict(), os.path.join(save_dir,"model_last.pth"))
        with torch.no_grad():
            for i, batch in enumerate(gene_loader):
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device)
                ret = model(batch,'generate')
                if test:
                    print('ret:',repr(ret))
                else:
                    ret_dict[batch['vn'][0]] = ret
        if not test:
            json_path = os.path.join(save_dir, str(epoch) + ".json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(ret_dict, f, ensure_ascii=False, indent=4)

def robust():

    frame = 8
    vocab_size = 50257
    log_dir = "logs"
    img_type = "salt"
    date_dir = img_type
    save_dir = os.path.join(log_dir, date_dir)
    os.makedirs(save_dir, exist_ok=True)
    # Translated comment.
    logging.info("begin training")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gpt2_model_path = r"E:\pycharmProject\clipcap\pretrained\gpt2"
    # Translated comment.
    model,tokenizer = define_feature_model(gpt2_model_path,device)
    model.to(device)

    max_len = 60
    test_dataset = Panda_feature('dataset\\panda',img_type, 'test_data.csv',
                                          caption_json="20250808.json", tokenizer=tokenizer,cap_type='short', max_len=max_len,frame=frame)
    gene_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    for epoch in range(2):
        ret_dict = {}
        if epoch == 0:
            model.load_state_dict(torch.load("logs/our_panda/model_best.pth"))
        else:
            model.load_state_dict(torch.load("logs/our_panda/model_last.pth"))
        with torch.no_grad():
            for i, batch in enumerate(gene_loader):
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device)
                ret = model(batch,'generate')
                ret_dict[batch['vn'][0]] = ret
            json_path = os.path.join(save_dir, str(epoch) + ".json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(ret_dict, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    train()

# ViTT
