function [x,A_]=AMR_Operator(A,i)
x=A(i,:);
A_=A;
A_(i,:)=0;
end