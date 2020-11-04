%% Baseline for correctness with python implementation
% Custom distance replicating sqEuclidean is used since Matlab force
% linkage type being 'ward' when pdist attribute is 'sqEuclidean'.
% Conversely when Distance is a function handle, linkage is set to 'ave'
% average linkage.

f = @customSqEuc;
load('X.mat')
%%

objGap=evalclusters(x,'linkage','gap','Distance',f,'B',10,'ReferenceDistribution','uniform','KList',[1:20]);
objGap.plot;
%%
function z = customSqEuc(x,y)
z = sum((x-y).*(x-y),2);
end