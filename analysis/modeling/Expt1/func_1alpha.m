function ll = func_1alpha(params,choice,reward,beta_prior)

alpha = params(1);
beta = params(2);

na = 2;

ntrials = length(choice);

q = ones(na,1)*1/na;
p = nan(1,ntrials);

% previous action
% pa = zeros(na,1);

for t = 1:ntrials
    if ~isnan(reward(t))  % valid trial


        %% policy
        pol = mcdougle_softmax_func(q,beta);
        p(t) = pol(choice(t));

        %% Q-learning
        q(choice(t)) = q(choice(t)) + alpha*(reward(t)-q(choice(t)));

    end

end

% underflow
epsilon=0.00000001;
p = epsilon/2 + (1-epsilon)*p;

% penalize ll based on prior on temperature param? (Leong, Radelescu et al. 2017)
if ~beta_prior
    ll = -nansum(log(p));
else
    ll = -nansum(log(p)) - log(gampdf(beta,2,3));
end













