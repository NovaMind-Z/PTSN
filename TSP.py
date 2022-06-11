from sklearn.cluster import KMeans
import numpy as np
import pickle
import torch
import nltk
import clip

N_A = ['JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'VB']

if __name__ == '__main__':

    def HClustering(concept_features, k):
        kmeans = KMeans(n_clusters=k, init='random').fit(concept_features)
        centers = kmeans.cluster_centers_  # [10, 512]
        hyper_labels = kmeans.labels_  # [7997,]
        return centers, hyper_labels

    # CLIP based word clustering
    vocab = pickle.load(open('./vocab.pkl', 'rb'))
    concepts = []
    for k in vocab.itos[4:]:
        pos_tag = nltk.pos_tag([k])[0][1]
        if pos_tag in N_A:
            concepts.append(k)
    print('concepts vocab build successfully, total length is', len(concepts))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=device)

    bsz = 100
    concept_features = []
    for i in range(0, len(concepts), bsz):
        text = concepts[i: i + bsz]
        text = clip.tokenize(text).to(device)
        with torch.no_grad():
            concept_feature = model.encode_text(text)
        concept_features.append(concept_feature.cpu().numpy())
    concept_features = np.vstack(concept_features)

    np.save('./concept_features.npy', concept_features)

    # It was used for 2000, 800, 400
    concept_features = np.load('./concept_features.npy')
    hyper_centers_dict  = torch.load('./hyper_centers.pth')
    centers, labels = HClustering(concept_features, 2000)
    hyper_centers_dict['hyper2k'] = torch.tensor(centers).float()
    torch.save(hyper_centers_dict, './hyper_centers.pth')

    hyper_vocab = {}
    for i, k in enumerate(concepts):
        hyper_vocab[k] = labels[i]
    torch.save(hyper_vocab, './hyper2k_vocab.pth')




    # CLIP based hierarchical clustering
    # 2000->800

    centers  = torch.load('./hyper_centers_.pth')['hyper2k']

    centers_2, labels_2 = HClustering(centers, 800)

    temp = torch.load('./hyper_centers.pth')
    temp['hyper2k-800'] =  torch.tensor(centers_2).float()
    torch.save(temp, './hyper_centers.pth')

    hyper_concepts = {}
    for k in range(len(centers)):
        hyper_id = labels_2[k]
        hyper_concepts[str(hyper_id)] = [k] if not str(hyper_id) in hyper_concepts else hyper_concepts[str(hyper_id)] + [k]
    torch.save(hyper_concepts, './hyper2k-800_concepts.pth')

    print('end')


