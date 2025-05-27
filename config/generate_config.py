import yaml

model, dataset = "BetaE", "CN15k"
template_path = f"config/train/{model}_{dataset}_soft.yaml"

with open(template_path, 'r') as file:
    yaml_data = yaml.safe_load(file)

cuda = 0
lr = 0.0001

omega_list = [0.5, 1.0, 1.5]
gamma = 1.0

for omega in omega_list:
    with open(template_path, 'r') as file:
        yaml_data = yaml.safe_load(file)
    yaml_data["train"]["learning_rate"] = lr
    yaml_data["estimator"]["beta"]["omega"] = omega
    yaml_data["estimator"]["beta"]["gamma"] = gamma
    out_path = f"config/search/{model}_{dataset}_soft_lr_{lr}_omega_{omega}_gamma_{gamma}.yaml"
    with open(out_path, 'w') as file:
        yaml_data = yaml.dump(yaml_data, file)
    print(omega)

cuda = 1
lr = 0.0005

omega_list = [0.5, 1.0, 1.5]
gamma = 1.0

for omega in omega_list:
    with open(template_path, 'r') as file:
        yaml_data = yaml.safe_load(file)
    yaml_data["train"]["learning_rate"] = lr
    yaml_data["estimator"]["beta"]["omega"] = omega
    yaml_data["estimator"]["beta"]["gamma"] = gamma
    out_path = f"config/search/{model}_{dataset}_soft_lr_{lr}_omega_{omega}_gamma_{gamma}.yaml"
    with open(out_path, 'w') as file:
        yaml_data = yaml.dump(yaml_data, file)
    print(omega)


cuda = 1