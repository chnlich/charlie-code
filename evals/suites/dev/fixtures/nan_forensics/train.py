from loss import cross_entropy

# A simulated training step. One logit is much larger than the others, which
# triggers the numerical instability in cross_entropy: the run crashes instead
# of printing a finite loss.
logits = [400.0, 1000.0, 600.0]
target = 0
loss = cross_entropy(logits, target)
print("final_loss=" + str(loss))
