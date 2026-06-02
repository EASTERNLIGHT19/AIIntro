import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 연산 장치 동적 할당

# CNN구조
class FashionCNN(nn.Module):
    
    def __init__(self, dropout_rate=0.25): 
        super(FashionCNN, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # 완전 연결층(FC)을 하나 더 추가
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        
        # 조절 가능한 드롭아웃 층 탑재
        self.dropout = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1) # 평탄화
        
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out) # 은닉층 통과 후 드롭아웃 적용
        out = self.fc2(out)
        return out


if __name__ == '__main__':
    print(f"연산 장치 셋업 완료: [{device}]")

    # [데이터 전처리] 
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5), 
        transforms.ToTensor()
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # PC에 저장된 Fashion MNIST 데이터셋 이용
    train_dataset = datasets.FashionMNIST(root='./dataset', train=True, transform=train_transform, download=True)
    test_dataset = datasets.FashionMNIST(root='./dataset', train=False, transform=test_transform, download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False)

    # 드롭아웃 비율 조절
    model = FashionCNN(dropout_rate=0.25).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # 최적화 기법(Adam) 탑재 & L2 정규화(weight_decay) 조절 기능
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)

    num_epochs = 30

    train_accuracies = []
    test_accuracies = []

    # ==========================================
    # 모델 학습 및 평가 루프
    # ==========================================
    print(" 학습 및 평가를 시작합니다.")
    for epoch in range(num_epochs):
        
        # [훈련 페이즈]
        model.train()
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        epoch_train_acc = 100 * correct_train / total_train
        train_accuracies.append(epoch_train_acc)

        # [테스트 페이즈]
        model.eval()
        correct_test = 0
        total_test = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        epoch_test_acc = 100 * correct_test / total_test
        test_accuracies.append(epoch_test_acc)

        # 콘솔 출력에서 정확도 표시
        print(f'Epoch [{epoch+1:2d}/{num_epochs}] Train Acc: {epoch_train_acc:.2f}% | Test Acc: {epoch_test_acc:.2f}%')

    # ==========================================
    # '정확도 변화 추이' 그래프 출력
    # ==========================================
    print("\n 학습 완료!, 정확도 추이 그래프를 화면에 출력합니다.")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs+1), train_accuracies, label='Train Accuracy', color='blue', marker='o')
    plt.plot(range(1, num_epochs+1), test_accuracies, label='Test Accuracy', color='red', marker='x')
    
    plt.title('Fashion MNIST Accuracy Trend')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
