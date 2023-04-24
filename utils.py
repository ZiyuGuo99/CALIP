import torch
import clip

def text_encode(classnames, templates, model):
    with torch.no_grad():
        text_feat = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates]  # format with class
            texts = clip.tokenize(texts).cuda()  # tokenize
            class_embeddings = model.encode_text(texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            text_feat.append(class_embedding)
        text_feat = torch.stack(text_feat, dim=1).cuda()
    return text_feat

def accuracy(output, label, n, topk=(1, 5)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(label.view(1, -1).expand_as(pred))
    return (100 * float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) / n for k in topk)

