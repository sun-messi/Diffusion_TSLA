# %%
# 基础包
# nohup python /home/sunj11/Documents/ECSE6966/Hw02/test1.py > output.log 2>&1 &
# # 查看输出日志
# tail -f output.log

# # 查看进程
# ps aux | grep python 

# # 监控GPU使用
# watch -n 1 nvidia-smi

import os
import numpy as np
import matplotlib.pyplot as plt

# PyTorch 相关
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, ConcatDataset

# 图像处理
from PIL import Image
from torchvision import transforms

# 进度条
from tqdm import tqdm

# 数学计算
import math


class TeslaDataset:
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform or transforms.ToTensor()
        
        if train:
            # 训练集：包含所有Tesla车型目录
            self.img_dirs = [
                os.path.join(root, 'Hw02/TSLA/New folder (2)/Cyber_Truck_processed1/'),
                os.path.join(root, 'Hw02/TSLA/New folder (2)/Model_E_processed1/'),
                os.path.join(root, 'Hw02/TSLA/New folder (2)/Model_S_processed1'),
                os.path.join(root, 'Hw02/TSLA/New folder (2)/Model_X_processed1'),
                os.path.join(root, 'Hw02/TSLA/New folder (2)/Model_Y_processed1')
                # os.path.join(root, 'Dataset_TSLA/Cybertruck_processed_aware'),
                # os.path.join(root, 'Dataset_TSLA/Model_3_processed_aware'),
                # os.path.join(root, 'Dataset_TSLA/Model_S_processed_aware'),
                # os.path.join(root, 'Dataset_TSLA/Model_X_processed_aware'),
                # os.path.join(root, 'Dataset_TSLA/Model_Y_processed_aware'),
            ]
        else:
            # 测试集：使用test_TSLA_128目录
            self.img_dirs = [os.path.join(root, 'test_TSLA_128')]
        
        # 收集所有图片路径
        self.images = []
        for img_dir in self.img_dirs:
            if os.path.exists(img_dir):
                files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                self.images.extend([(os.path.join(img_dir, f)) for f in files])
        
        if len(self.images) == 0:
            raise ValueError(f"No images found in the specified directories")
        
        self.images = sorted(self.images)
        print(f"Found {len(self.images)} images in {'training' if train else 'testing'} set")
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, 0  # 返回图片和虚拟标签0
    
    def __len__(self):
        return len(self.images)

def show_images(dataset, num_samples=5, cols=5):
    # 获取数据集大小
    dataset_size = len(dataset)
    num_samples = min(num_samples, dataset_size)
    if num_samples == 0:
        return
    
    # 生成随机索引
    random_indices = torch.randperm(dataset_size)[:num_samples]
    
    rows = (num_samples + cols - 1) // cols
    plt.figure(figsize=(3*cols, 3*rows))
    
    for i in range(num_samples):
        # 使用随机索引获取图片
        img, _ = dataset[random_indices[i]]
        plt.subplot(rows, cols, i + 1)
        
        if isinstance(img, torch.Tensor):
            img_np = img.permute(1, 2, 0).numpy()
            img_np = np.clip(img_np, 0, 1)
        else:
            img_np = np.array(img)
        
        plt.imshow(img_np)
        plt.axis('off')
        plt.title(f'Image {random_indices[i].item()}')  # 显示实际的图片索引
    
    plt.tight_layout()
    plt.show()

def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

def get_index_from_list(vals, t, x_shape):
    """
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def forward_diffusion_sample(x_0, t, device="cpu"):
    """
    Takes an image and a timestep as input and
    returns the noisy version of it
    """
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    # mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
    + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)

class ResizeWithPad:
    def __init__(self, size):
        self.size = size
        # ImageNet 均值 (R,G,B)
        self.fill = (123, 116, 103)  # ImageNet mean values

    def __call__(self, img):
        w, h = img.size
        ratio = min(self.size/h, self.size/w)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        
        # 按比例缩放
        resized = transforms.Resize((new_h, new_w))(img)
        
        # 使用 ImageNet 均值创建背景
        new_img = Image.new('RGB', (self.size, self.size), self.fill)
        
        # 居中粘贴
        paste_x = (self.size - new_w) // 2
        paste_y = (self.size - new_h) // 2
        new_img.paste(resized, (paste_x, paste_y))
        
        return new_img

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

def load_transformed_dataset():
    data_transforms = [
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, IMG_SIZE)),  # 使用中心裁剪
        transforms.RandomHorizontalFlip(),  # 随机水平翻转（数据增强）
        transforms.ToTensor(),  # 转为 PyTorch Tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        # transforms.Lambda(lambda t: (t * 2) - 1)  # 归一化到 [-1, 1]
    ]
    data_transform = transforms.Compose(data_transforms)

    # 使用我们自定义的 StanfordCars 类
    train = TeslaDataset(root="/home/sunj11/Documents/ECSE6966", 
                        train=True,
                        transform=data_transform)

    test = TeslaDataset(root="/home/sunj11/Documents/ECSE6966", 
                       train=False,
                       transform=data_transform)
    
    return ConcatDataset([train, test])

def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    plt.imshow(reverse_transforms(image))

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp =  nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()

    def forward(self, x, t, ):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # Add time channel
        h = h + time_emb
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # TODO: Double check the ordering here
        return embeddings


class SimpleUnet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """
    def __init__(self):
        super().__init__()
        image_channels = 3
        down_channels = (32, 64, 128, 256, 512, 1024, 2048)
        up_channels = (2048, 1024, 512, 256, 128, 64, 32)
        # down_channels = (64, 128, 256, 512)
        # up_channels = (512, 256, 128, 64)
        # 原始通道数除以4并转换为整数
        # down_channels = (16, 32, 64, 128, 256)  
        # up_channels = (256, 128, 64, 32, 16)
        out_dim = 3
        time_emb_dim = 32

        # Time embedding
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.ReLU()
            )

        # Initial projection
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        # Downsample
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1], \
                                    time_emb_dim) \
                    for i in range(len(down_channels)-1)])
        # Upsample
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], \
                                        time_emb_dim, up=True) \
                    for i in range(len(up_channels)-1)])

        # Edit: Corrected a bug found by Jakub C (see YouTube comment)
        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, timestep):
        # Embedd time
        t = self.time_mlp(timestep)
        # Initial conv
        x = self.conv0(x)
        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)
        return self.output(x)


def get_loss(model, x_0, t):
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    noise_pred = model(x_noisy, t)
    return F.l1_loss(noise, noise_pred)

@torch.no_grad()
def sample_timestep(x, t):
    """
    Calls the model to predict the noise in the image and returns
    the denoised image.
    Applies noise to this image, if we are not in the last step yet.
    """
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

    if t == 0:
        # As pointed out by Luis Pereira (see YouTube comment)
        # The t's are offset from the t's in the paper
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def sample_plot_image():
    # Sample noise
    img_size = IMG_SIZE
    img = torch.randn((1, 3, img_size, img_size), device=device)
    plt.figure(figsize=(15,15))
    plt.axis('off')
    num_images = 10
    stepsize = int(T/num_images)

    for i in range(0,T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(img, t)
        # Edit: This is to maintain the natural range of the distribution
        img = torch.clamp(img, -1.0, 1.0)
        if i % stepsize == 0:
            plt.subplot(1, num_images, int(i/stepsize)+1)
            show_tensor_image(img.detach().cpu())
    plt.show()
    
    
def get_loss(model, x_0, t):
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    noise_pred = model(x_noisy, t)
    return F.l1_loss(noise, noise_pred)

def save_checkpoint(epoch, model, optimizer, loss, save_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, save_path)
    print(f"Checkpoint saved: {save_path}")
   

def show_and_save_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    transformed_image = reverse_transforms(image)
    return transformed_image  # 返回转换后的PIL Image


@torch.no_grad()
def sample_and_save_image(save_dir, epoch):
    os.makedirs(save_dir, exist_ok=True)
    
    img_size = IMG_SIZE
    img = torch.randn((1, 3, img_size, img_size), device=device)
    
    plt.figure(figsize=(15,15))
    plt.axis('off')
    num_images = 10
    stepsize = int(T/num_images)
    
    generated_images = []

    for i in range(0,T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(img, t)
        img = torch.clamp(img, -1.0, 1.0)
        
        if i % stepsize == 0:
            generated_images.append(img.detach().cpu())
            plt.subplot(1, num_images, int(i/stepsize)+1)
            transformed_img = show_and_save_tensor_image(img.detach().cpu())
            plt.imshow(transformed_img)
    
    plt.savefig(os.path.join(save_dir, f'diffusion_process_epoch_{epoch}.png'))
    plt.close()
    
    # 保存最终生成的图像
    final_image = generated_images[-1]
    transformed_image = show_and_save_tensor_image(final_image)
    save_path = os.path.join(save_dir, f'generated_image_epoch_{epoch}.png')
    transformed_image.save(save_path)
    print(f"Images saved for epoch {epoch} in {save_dir}")
    
    return final_image
   
    


# %%
# 基础参数设置
IMG_SIZE = 256
BATCH_SIZE = 128
T = 1000  # 时间步数
base_dir = "/home/sunj11/Documents/ECSE6966"

# 创建保存目录
save_dir = 'checkpoints_scratch_new_square256_T1000'
os.makedirs(save_dir, exist_ok=True)

# 1. 数据集准备
train_dataset = TeslaDataset(root=base_dir, train=True)
test_dataset = TeslaDataset(root=base_dir, train=False)

# 显示数据集样例
print("\nShowing training images:")
show_images(train_dataset, num_samples=10, cols=5)
print("\nShowing testing images:")
show_images(test_dataset, num_samples=10, cols=5)

# 2. 模型预处理
# 计算扩散过程参数
betas = linear_beta_schedule(timesteps=T)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

# 3. 数据加载器设置
data = load_transformed_dataset()
dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=32)

# %%
# 4. 模型初始化和GPU设置
model = SimpleUnet()
print("Num params: ", sum(p.numel() for p in model.parameters()))

# GPU设置
num_gpus = torch.cuda.device_count()
print(f"Number of GPUs available: {num_gpus}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 将模型移动到GPU
model = model.to(device)
if num_gpus > 1:
    device_ids = list(range(num_gpus))
    model = torch.nn.DataParallel(model, device_ids=device_ids)
print(f"Model is using GPUs: {device_ids}")

# 5. 训练设置
lr = 1e-3
optimizer = Adam(model.parameters(), lr=lr)
epochs = 500
best_loss = float('inf')

# 6. 训练循环
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    num_batches = 0
    
    for step, batch in enumerate(dataloader):
        # 清空梯度
        optimizer.zero_grad()
        
        # 准备数据
        images = batch[0].to(device)
        t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
        
        # 前向传播和反向传播
        loss = get_loss(model, images, t)
        loss.backward()
        optimizer.step()
        
        # 记录损失
        epoch_loss += loss.item()
        num_batches += 1

        # 打印训练进度
        if epoch % 1 == 0 and step == 0:
            print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item():.6f} ")
        
        # 定期保存和采样
        if epoch % 10 == 0 and step == 0:
            # 保存检查点
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
            save_checkpoint(epoch, model, optimizer, loss, checkpoint_path)
            
            # 生成样本图片
            with torch.no_grad():
                sample_and_save_image(os.path.join(save_dir, 'generated_images'), epoch) 
            
            # 保存最佳模型
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_model_path = os.path.join(save_dir, 'best_model.pth')
                save_checkpoint(epoch, model, optimizer, loss, best_model_path)
        
        # 学习率调整
        if epoch % 200 == 0 and step == 0:
            lr = lr * 0.5
            optimizer = Adam(model.parameters(), lr=lr)

    # 打印epoch平均损失
    avg_loss = epoch_loss / num_batches
    print(f"Epoch {epoch} Average Loss: {avg_loss:.6f}")

# 保存最终模型
final_checkpoint_path = os.path.join(save_dir, 'final_model.pth')
save_checkpoint(epochs-1, model, optimizer, loss, final_checkpoint_path)



# %%



