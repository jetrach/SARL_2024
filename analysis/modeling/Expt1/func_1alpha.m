function ll = func_1alpha(params,choice,reward,beta_prior,pairAssignment)

% randomly initialized parameters
alpha = params(1);
beta = params(2);
% how many options per trial
na = 2;
% how many trials
ntrials = length(choice);

% initialize q values 2 columns (choices per trial) and 4 rows (n bandit
% pairs)
q = ones(na,4)*1/na;
p = nan(1,ntrials);

for t = 1:ntrials
    if ~isnan(reward(t))  % valid trial

        [r,c] = find(pairAssignment == choice(t));
        %% policy
        pol = mcdougle_softmax_func(q(r,:),beta);
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













