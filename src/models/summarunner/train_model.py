import torch
import time
import torch.optim as optim
import torch.nn as nn


def train_model(model, train_iterator, val_iterator, vocabulary, bpe_processor,
                epoch_count=2, loss_every_nsteps=16, lr=0.001, device_name="cuda"):
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {param_count}")

    device = torch.device(device_name)
    model = model.to(device)

    total_loss = 0
    start_time = time.time()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.BCEWithLogitsLoss().to(device)

    for epoch in range(epoch_count):
        for step, batch in enumerate(train_iterator):

            model.train()
            logits = model(batch["inputs"])
            loss = loss_function(logits, batch["outputs"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if step % loss_every_nsteps == 0 and step != 0:
                val_total_loss = 0
                val_batch_count = 0

                model.eval()
                for _, val_batch in enumerate(val_iterator):
                    # print(val_batch["inputs"].shape, batch['outputs'].shape)
                    logits = model(val_batch["inputs"])
                    val_total_loss += loss_function(logits, batch['outputs'])
                    val_batch_count += 1

                avg_val_loss = val_total_loss/val_batch_count
                print(f"Epoch = {epoch}, Avg train loss: {total_loss/loss_every_nsteps}, Avg val loss: {avg_val_loss}, Time: {start_time}")

                total_loss = 0
                start_time = time.time()
        total_loss = 0
        start_time = time.time()