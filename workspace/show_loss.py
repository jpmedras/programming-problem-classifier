import matplotlib.pyplot as plt

def show_loss_evolution(num_epochs, train_losses, test_losses):
    plt.plot(range(1, num_epochs + 1), train_losses, marker='o', linestyle='-', color='b', label='Train')
    plt.plot(range(1, num_epochs + 1), test_losses, marker='x', linestyle='-', color='g', label='Test')

    plt.title('Learning Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Average Loss')
    plt.legend()
    plt.show()