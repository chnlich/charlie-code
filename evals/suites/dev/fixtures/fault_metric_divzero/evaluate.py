from metrics import precision, recall, f1_score

# Normal case
print("precision:%s" % precision(10, 5, 3, 20))
print("recall:%s" % recall(10, 5, 3, 20))
print("f1:%s" % f1_score(10, 5, 3, 20))

# Edge case: no positive predictions (tp=0, fp=0)
print("precision_edge:%s" % precision(0, 0, 5, 20))
print("recall_edge:%s" % recall(0, 0, 5, 20))
print("f1_edge:%s" % f1_score(0, 0, 5, 20))

# Edge case: no positive ground truth (tp=0, fn=0)
print("precision_edge2:%s" % precision(0, 5, 0, 20))
print("recall_edge2:%s" % recall(0, 5, 0, 20))
print("f1_edge2:%s" % f1_score(0, 5, 0, 20))
