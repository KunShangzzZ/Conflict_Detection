function [x_new]=Max_Operator(x,N)
%{
 Notice x must be a vector 
 N means the first N  maximums value in x, which we want to reserve.
%}
x_new=zeros(size(x));
[~,index]=sort(x,'descend'); 
Index=index(1:N);
x_new(Index)=x(Index);
end