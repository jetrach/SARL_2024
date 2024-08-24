%% Simulation code for Experiment 
%% 12/3/2020; New Haven, CT

clear all;close all;clc;

% load data
data = readtable('SARLrep_prolific_n133_final.csv');
subs = unique(data.subject);
nsubs = length(subs);

% load fit
load models/modelFit_2a
mod = modelFit_2a;

mean_sim_r = nan(nsubs,76);
mean_sim_cor = nan(nsubs,76);
mean_sim_a = nan(nsubs,76);

mean_sub_r = nan(nsubs,76);
mean_sub_cor = nan(nsubs,76);
mean_sub_a = nan(nsubs,76);

for si = 1:nsubs

    subidx = strcmp(data.subject,subs{si});
    subdata = data(subidx,:);

    % get participant group
    group = unique(subdata.group);

    preRLidx = strcmp(subdata.rlType,'preRL');
    attdata = subdata(preRLidx,:);

    % all SA trials
    attalldata = subdata(:,:);

    RLidx = strcmp(subdata.trialType,'rl');
    rldata = subdata(RLidx,:);

    bad =  strcmp(rldata.reward,'NA');

    reward = rldata.reward;
    reward = str2double(reward); % format as vector
    reward(isnan(rldata.key_press)) = nan; % NaN the bad trials


    choiceBandit = rldata.bandChosen;
    nrl = length(choiceBandit);
    choice = nan(1,length(reward));
    bandit_probs = nan(2,length(reward));
    type = nan(1,nrl);cor = nan(1,nrl);rt = nan(1,nrl);


    ntrials = length(choice);sub_cor=[];

    % set this up so that the SAME SCHED is in the same place for each subj
    % (not by shape)
    if group == 1
        bandit_probs(1,:) = str2double(rldata.bandOneVal);
        bandit_probs(2,:) = str2double(rldata.bandTwoVal);
    elseif group == 2
        bandit_probs(1,:) = str2double(rldata.bandTwoVal);
        bandit_probs(2,:) = str2double(rldata.bandOneVal);
    end
    % compute RT attention proxy
    for t = 1:ntrials
        if group == 1
            if strcmp(choiceBandit{t},'74')
                choice(t) = 1;
                sub_cor(t) = bandit_probs(1,t);
                sub_a(t) = 1;
            elseif strcmp(choiceBandit{t},'75')
                choice(t) = 2;
                sub_cor(t) = bandit_probs(2,t);
                sub_a(t) = 0;
            else
                choice(t) = nan;
                sub_cor(t) = nan;
                sub_a(t) = nan;
            end
        elseif group == 2
            if strcmp(choiceBandit{t},'75')
                choice(t) = 1;
                sub_cor(t) = bandit_probs(1,t);
                sub_a(t) = 1;
            elseif strcmp(choiceBandit{t},'74')
                choice(t) = 2;
                sub_cor(t) = bandit_probs(2,t);
                sub_a(t) = 0;
            else
                choice(t) = nan;
                sub_cor(t) = nan;
                sub_a(t) = nan;
            end
        end
    end

    % model
    alpha = mod.alpha(si);
    alphaneg = mod.alphaneg(si);
    beta = mod.beta(si);

    na = 2;

    nsims = 100;
    sim_r = nan(nsims,ntrials);
    sim_a = nan(nsims,ntrials);

    for sim = 1:nsims

        q = ones(na,1)*(1/na);
        simrt = rt(randperm(length(rt)));
        pa = zeros(1,2)';

        for n = 1:ntrials

            % choose bandit
            pol = (exp(q.*beta)./sum(exp(q.*beta)))';

            % % Now select a machine
            x = rand;
            counts = histc(x,[0,cumsum(pol)]); % Setting up bins of prob intervals, which one is rand number "x" in?
            a = find(counts==1); % action=find bins that contain x

            % was agent rewarded?
            sim_r(sim,n) = bandit_probs(a,n);

            % did they choose a?
            if a == 1
                sim_a(sim,n) = 1;
            elseif a == 2
                sim_a(sim,n) = 0;
            end

            % was agent correct?
            if bandit_probs(a,n) > bandit_probs(3-a,n)
                sim_cor(sim,n) = 1;
            else
                sim_cor(sim,n) = 0;
            end

            if sim_r(sim,n) <= 0
                lr = alphaneg;
            else
                lr = alpha;
            end

            %% Q-learning
            q(a) = q(a) + lr*(sim_r(sim,n)-q(a));


            %         % update previous action vector for stickyness
            pa(:) = 0;
            pa(a) = 1;

        end
    end
    mean_sim_r(si,1:ntrials) = nanmean(sim_r);
    mean_sim_cor(si,1:ntrials) = nanmean(sim_cor(:,1:ntrials));
    mean_sim_a(si,1:ntrials) = nanmean(sim_a(:,1:ntrials));

    mean_sub_r(si,1:ntrials) = reward;
    mean_sub_cor(si,1:ntrials) = sub_cor;
    mean_sub_a(si,1:ntrials) = sub_a;

end


figure;
plot(nanmean(mean_sub_a),'-ok');hold on;
plot(nanmean(mean_sim_a),'r');
title('rewards');
xlabel('trial');ylabel('p(choose A)');

% Set subplot dim
r = 5;
c = 5;
for sub = 1:133
    subplotNum = floor(sub/(r*c));
    plotNum = sub-(r*c*subplotNum);
    if(plotNum == 0)
        subplot(r,c, (r*c)); % Create a subplot in a 7x5 grid
    else
        if plotNum == 1
            figure;
        end
        subplot(r,c, (sub-(r*c*subplotNum))); % Create a subplot in a 7x5 grid
    end


    plot(nanmean(mean_sub_a),'-ok','MarkerFaceColor','k','linewidth',2,'markersize',5); hold on;
    plot(mean_sim_a(sub,:)','color',[.8 .3 .1],'linewidth',2);
    xlabel('trial'); ylabel('p(correct)');
    %legend('data','model','reversal','location','southwest');
    title(sub)
    box off;
    set(gca,'tickdir','out','linewidth',2)
end
% Adjust the figure size if needed
set(gcf,'position',[5 613 1200 800]); % Adjust the figure size to accommodate the subplots
print -dtiff -r300 SARL_ntb_lab_model

figure;
plot(normalize(bandit_probs(1,:),'range'),'-ok');hold on;
plot(nanmean(mean_sim_a),'r');
title('pChoose bandit A model (black scaled reward sched)');
xlabel('trial');ylabel('p(choose A)');

figure;
plot(normalize(bandit_probs(2,:),'range'),'-ok');hold on;
plot(nanmean(mean_sim_a),'r');
title('pChoose bandit A model (black scaled reward sched)');
xlabel('trial');ylabel('p(choose b)');

figure;
plot(bandit_probs(1,:),'b');hold on;
plot(bandit_probs(2,:),'r');hold on;
title('reward schedules');
legend('band A', 'band B')
xlabel('trial');ylabel('points');





