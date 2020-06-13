import torch

def fgsm(model, X, y, epsilon):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    loss = nn.CrossEntropyLoss()(model(X + delta), y)
    loss.backward()
    return epsilon * delta.grad.detach().sign()


def pgd_linf(model, X, y, epsilon, alpha, num_iter):
    """ Construct PGD adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    for t in range(num_iter):
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    return delta.detach()



def test_attack(model, X, y, attack):
    delta = attack(model, X, y, 0.1)
    predictions = model(X + delta)

	#test visualisation
	M, N = 2, 6
	images = test_x[0:M*N].detach().cpu()
	labels = [letters[test_y[i].detach().cpu()] for i in range(M*N)]
	predictions = [letters[torch.max(predictions.data, 1)[1][i].item()] for i in range(M*N)]

	visualisation.log_image_grid(images, labels, predictions, M, N, writer)