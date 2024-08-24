
function [ll] = func_2alpha(params,choice,reward,beta_prior)

alpha = params(1); % alpha
alphaneg = params(2);  % alpha neg
beta = params(3);  % beta

na = 2; % number of available actions

ntrials = length(choice); % N trials

q = ones(na,1)*1/na; % init q values
p = nan(1,ntrials); % init vector of probablities

% main loop
for t = 1:ntrials
    if ~isnan(reward(t))  % valid trial

        if reward(t) == 1
            lr = alpha; % pos alpha
        elseif reward(t) == 0
            lr = alphaneg; % neg alpha
        end

        %% policy
        pol = mcdougle_softmax_func(q,beta); % compute policy using softmax
        p(t) = pol(choice(t)); % p is the computed policy value for the actual choice the subject made

        %% Q-learning!
        q(choice(t)) = q(choice(t)) + lr*(reward(t)-q(choice(t))); % update q value based on choice/outcome
    end

end
% underflow to prevent some weird stuff
epsilon=0.00000001;
p = epsilon/2 + (1-epsilon)*p;

%% here's where you imlement the prior and compute the total neg log liklihood
% penalize ll based on prior on temperature param? (Leong, Radelescu et al. 2017)
if ~beta_prior
    ll = -nansum(log(p));
else
    ll = -nansum(log(p)) - log(gampdf(beta,2,3));
end













