% function to compute the population count distribution p(k) from a set of samples

function pk = samplepk(samples)
[N, n_samples] = size(samples);
counts = sum(samples, 1); % how many neurons fire in each state
ks = unique(counts);
k_freq = hist(counts, ks); % how often each count pops up
pk = zeros(1, N+1);
for i = 1:length(ks)
    pk(ks(i)+1)= k_freq(i);
end
pk = pk / n_samples;
end