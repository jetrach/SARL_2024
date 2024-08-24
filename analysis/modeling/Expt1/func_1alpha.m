
function ll = func_1a(params,choice,reward,attention,rt,beta_prior)

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
%         pol = noise*(1/na) + (1-noise)*pol; % undirected noise
        p(t) = pol(choice(t));
        
        %% Q-learning
        q(choice(t)) = q(choice(t)) + alpha*(reward(t)-q(choice(t)));
        
%         %% counterfactual?
%         q(3-choice(t)) = q(3-choice(t)) + alphaC*((1-reward(t))-q(3-choice(t)));        
        
        % update previous action vector for stickyness
%         pa(:) = 0;
%         pa(choice(t)) = 1;
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













