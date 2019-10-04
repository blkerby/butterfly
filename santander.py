import pandas as pd
import torch
import torch.nn.functional
import numpy as np
from butterfly import CustomNetwork
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

dtype = torch.float
seed = 0
device = torch.device('cuda')
batch_size = 16384

torch.random.manual_seed(seed)

def compute_loss(pred, y):
    return torch.nn.functional.binary_cross_entropy_with_logits(pred, y)

def compute_objective(model, loss):
    return loss + model.penalty()

df_full = pd.read_csv("train.csv") #.iloc[:10000, :]
y_full = df_full['target'].values
X_full = df_full.iloc[:, 2:].values
X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=0)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(np.ascontiguousarray(X_train.transpose()), dtype=dtype, device=device)
X_test = torch.tensor(np.ascontiguousarray(X_test.transpose()), dtype=dtype, device=device)
y_train = torch.tensor(y_train, dtype=dtype, device=device)
y_test = torch.tensor(y_test, dtype=dtype, device=device)

# Construct/initialize the model
model = CustomNetwork(
    num_inputs=X_train.shape[0],
    num_outputs=1,
    width_pow=10,
    depth=4,
    butterfly_depth=10,
    l2_slope=0.1, # 0.00003, #0.0000005, #0.0001,
    # l2_slope=0.000001, #0.0001,
    l2_scale=1e-6, #1e-7, #1e-5, #1e-4, #2e-4, #0.0000001, # 0.0000001,#0.00001,
    l2_bias=0.0,
    l2_interact=0.0,
    dtype=dtype,
    device=device
)

print("model: width_pow={}, depth={}, l2_slope={}, l2_scale={}, l2_interact={}".format(
    model.width_pow, model.depth, model.l2_slope, model.l2_scale, model.l2_interact
))

optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=1, max_eval=20, history_size=500, tolerance_grad=0, tolerance_change=0,
                              line_search_fn='strong_wolfe')


y_test_cpu = y_test.cpu()
print("y_test_cpu: {}".format(y_test_cpu.shape))
last_loss = float("Inf")
last_gn = float("Inf")
for i in range(100000):
    eval_cnt = 0

    def closure():
        global eval_cnt
        global pred
        global train_loss
        global obj
        eval_cnt += 1
        optimizer.zero_grad()
        model.zero_grad()
        obj = 0
        train_loss = 0
        j = 0
        while j < X_train.shape[1]:
            X_train_batch = X_train[:, j:(j + batch_size)]
            y_train_batch = y_train[j:(j + batch_size)]
            pred = model(X_train_batch)[0, :]
            batch_train_loss = compute_loss(pred, y_train_batch)
            batch_obj = compute_objective(model, batch_train_loss)
            batch_obj.backward()
            obj += float(batch_obj) * len(y_train_batch)
            train_loss += float(batch_train_loss) * len(y_train_batch)
            j += batch_size
        obj /= len(y_train)
        train_loss /= len(y_train)
        return obj

    optimizer.step(closure)

    if i % 1 == 0:
        with torch.no_grad():
            # print(list(model.parameters()))
            gn = torch.sqrt(sum(torch.sum(X.grad**2) for X in model.parameters()))
            pred_test = torch.zeros_like(X_test[0, :])
            j = 0
            while j < X_test.shape[1]:
                X_test_batch = X_test[:, j:(j + batch_size)]
                pred_test_batch = model(X_test_batch)
                pred_test[j:(j + batch_size)] = pred_test_batch
                j += batch_size
            
            test_loss = compute_loss(pred_test, y_test)
            test_auc = roc_auc_score(y_test_cpu, pred_test.cpu())
            print("seed {}, iteration {}: obj {:.7f}, train {:.7f}, test loss {:.7f}, test auc {:.7f}, obj grad norm {:.7f}".format(
                seed, i, float(obj), float(train_loss), float(test_loss), test_auc, gn))
            if last_loss == train_loss and last_gn == gn:
                break
            last_loss = train_loss
            last_gn = gn
