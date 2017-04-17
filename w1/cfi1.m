% Const Function Intuition I 
% Draw graphs of hypothesis function and cost function
% Training set = (1,1), (2,2), (3,3)

% clean environment
clc; close all; clear;

% training data
X = [1 2 3]';
Y = [1 2 3]';

theta1 = [0.5, 1, 0]; % scattered theta1
n = size(theta1, 2);

fig = figure;
gh = subplot(1, 2, 1);
gj = subplot(1, 2, 2);

% plot training data
scatter(gh, X, Y, 'LineWidth', 1.5);

xlabel(gh, '\boldmath$x$', 'Interpreter', 'latex');
ylabel(gh, '\boldmath$y$', 'Interpreter', 'latex');
title(gh, '\boldmath$h_{\theta}(x)$', 'Interpreter', 'latex');
xlim(gh, [0 6]); ylim(gh, [0 4]);

I = (0:6)';
theta0 = 0; % for simplify

% plot hypothesis lines
for i = 1:n
    theta = [theta0; theta1(i)];
    H = hypothesis(I, theta);
    subplot(1, 2, 1); hold on;
    plot(gh, I, H, 'LineWidth', 1.5);
end

labels = cell(n+1,1);
labels{1} = 'training data';
for i = 2:(n+1)
    labels{i} = sprintf('\\theta_1 = %g', theta1(i-1));
end
legend(gh, labels, 'Location', 'northwest');

% plot cost function
t = (-1:0.1:3);
m = size(t, 2);
J = zeros(m,1);
for i = 1:m
    theta = [theta0; t(i)];
    J(i) = getJ(X, Y, theta);
end
plot(gj, t, J);

xlabel(gj, '\boldmath$\theta_{1}$', 'Interpreter', 'latex');
ylabel(gj, '\boldmath$J$', 'Interpreter', 'latex');
title(gj, '\boldmath$J(\theta_1)$', 'Interpreter', 'latex');
xlim(gj, [-1 3]); ylim(gj, [0 10]);

% scatter theta1
J = zeros(n, 1);
gt = gobjects(n, 1);
for i = 1:n
    theta = [theta0; theta1(i)];
    J(i) = getJ(X, Y, theta);
    subplot(1, 2, 2); hold on;
    gt(i) = plot(gj, theta1(i), J(i), 'x', 'LineWidth', 1.5);
end

legend(gj, gt, labels{2:(n+1)});

% set main title
ha = axes('Position', [0 0 1 1], 'Xlim', [0 1], 'Ylim', [0 1],...
    'Box', 'off', 'Visible', 'off', 'Units', 'normalized', 'clipping' , 'off');
text(0.5, 1, '\bf Cost Function Intuition I', 'HorizontalAlignment',...
    'center', 'VerticalAlignment', 'top');

% save figure to file
% fig.PaperType = 'a4';
% fig.PaperOrientation = 'landscape';
% print('cfi1','-dpdf','-fillpage');