from bovw.bovw import BOVW

dataset = [
    "example_dataset/img1.jpg",
    "example_dataset/img2.jpg",
    "example_dataset/img3.jpg",
]

model = BOVW(n_clusters=10)
model.train(dataset)
