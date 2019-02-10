clc; clear; close all; currpath = cd; addpath(genpath(currpath));
%% --- loading ---
load 'data2.mat' %t=0原始矩阵
[nAI,NUM_DMs]=size(data); %NOTE THAT:Decision Makers-DMs、numerical Assessment Information-nAI
% --- Para ---
n=9;   %方案个数
N=3;   %属性个数
M=6;
m=30;
alpha=0.3;  %不和谐度阈值-NEED alpha
Tmax=7;    %最大循环次数
%Q=0.28;      %群冲突度阈值 -NEED theta
XI=[1/3,1/3,1/3]; % the weights of attributes
t=0;
E=1:m;  %输入决策者集合

u=0.4; % 计算更新方案时 原d_k的评价信息比例
r=1-u; % 计算更新方案时 与 d_k的结合的其他专家的评价信息比例
% the parameters of solve the SR problem
lambda=1.e-2;
tolerance=1.e-3;

%% Spares representation
% --- Phase 1 : Conflict Relationship Investigation ---
D=data; % 因为每次调整一列DD中的值 一直在变幻 所以用D替代变换过程中的
SR_coeffcients=zeros(NUM_DMs);
for m=1:NUM_DMs
    [y,data_]=AMR_Operator(data',m);
    [Coef, ~] = SolveHomotopy(data_',y','lambda',lambda,'tolerance',tolerance);
    SR_coeffcients(:,m)=Coef; %column
end
BETA=SR_coeffcients;
SR_coeffcients(SR_coeffcients>0)=0;
C=abs(SR_coeffcients);% Establish the conflict network




%% ---  Conflict relationship detection ---
% detect the opinion conflict or behavior conflict
% NOTE THAT OC means opinion conflict, BC means behavior conflict
OC=zeros(1,NUM_DMs);
BC=OC;
for iii=1:NUM_DMs
    BC(iii)=sum(C(:,iii));  %calculate \sum_j=1^I c(ij)
    OC(iii)=sum(C(iii,:));  %calculate \sum_j=1^I c(ji)
end


%% calculate AC and CD
AC=sum(OC)/NUM_DMs;
CD= 2*sum(sum(C(:,:)~=0))/(NUM_DMs*(NUM_DMs-1)); % group conflict degree CD
Q=alpha*AC;  % set the conflict threshold

Operation=5;%为了让程序进行
while (Operation==5||Operation==6&&t<=Tmax&&~isempty(E))
    SR_coeffcients=zeros(NUM_DMs);
    D=DD;
    for i=1:NUM_DMs
        y=D(:,i);
        D_=D;
        D_(:,i)=0;                      %AMR Operator
        beta=SR(D_,y);
        SR_coeffcients(:,i)=beta;       %column
    end
    BETA=SR_coeffcients;
    SR_coeffcients(SR_coeffcients>0)=0;
    C=abs(SR_coeffcients);
    CD= 2*sum(sum(C(:,:)~=0))/(NUM_DMs*(NUM_DMs-1)); % group conflict degree CD
    OC=zeros(1,NUM_DMs);
    BC=OC;
    for iii=1:NUM_DMs
        BC(iii)=sum(C(:,iii)); %calculate \sum_j=1^I c(ij)
        OC(iii)=sum(C(iii,:)); %calculate \sum_j=1^I c(ji)
    end
    %% 对决策者的冲突行为进行分类
    Upsilon=[];Omega=[]; Psi=[];
    for j=1:NUM_DMs
        if OC(j)==0
            if BC(j)==0
                Upsilon=[Upsilon j];
            else
                Psi=[Psi j];
            end
        else
            if BC(j)==0
                Omega=[Omega j];
            elseif OC(j)>BC(j)
                Omega=[Omega j];
            elseif OC(j)<BC(j)
                Psi=[Psi j];
            elseif OC(j)==BC(j)  %%Ding add
                if OC(j)<=AC;
                    Omega=[Omega j];
                else
                    Psi=[Psi j];
                end
            end
        end
    end
    
    CA=zeros(nAI,1); %collective opinion
    omega=zeros(NUM_DMs,1); %omega is the weight vector of DMs
    %%计算决策者的权重
    for iii=1:NUM_DMs
        omega(iii)=(1-OC(iii)/sum(OC)-BC(iii)/sum(BC))/(NUM_DMs-2); % compute every weight for each DM
    end
    %%决算全决策信息
    CA=D*omega;%这一步是获得collective assessment CA
    
    
    CAM=zeros(M,N);  % restructured CA
    S=zeros(M,1); % the score vector
    disp(['CD=',num2str(CD)])
    disp(['Theta=',num2str(Q)])
    E=[Omega Psi]; %E是Omega和Psi的并集
    %最后还没有加上对应的end
    if CD<=Q %需要再方案调整时更新存储p(t) D CA
        
        %disp(D)
        CAM=reshape(CA,M,N); %重构
        S=CAM*XI'; %获得方案的评价得分 这三步是在Section 5
        TTT=2*t;
        result(TT).Upsilon=Upsilon;
        result(TT).Omega=Omega;
        result(TT).Psi=Psi;
        result(TT).CA=CA;
        result(TT).S=S;
        result(TT).omega=omega;
        
        disp(['D=DD'])
        disp(['方案得分结果为：S=', num2str(S')])
        break
    else %pt>Q 没有达到冲突阈值时进行如下
        if t<=Tmax
            t=t+1;
            
            CAM=reshape(CA,M,N); %重构
            S=CAM*XI'; %获得方案的评价得分 这三步是在Section 5
            T=2*t-1;
            result(T).Upsilon=Upsilon;
            result(T).Omega=Omega;
            result(T).Psi=Psi;
            result(T).CA=CA;
            result(T).S=S;
            result(T).omega=omega;
            disp(['第',num2str(t),'次循环'])
            
            [q, a]=size(Omega); %求Omega里面DM的个数a
            [w,b]=size(Psi);%求Psi里面DM的个数b
            [V, z]=size(Upsilon); %求Upsilon 里面DM的个数
            AAA=[];B=[];AA=[];BB=[];J=[];II=[];
            OCC=[];%对应Omega集合里面最大BC值的DM的调整方案集合
            BCC=[];%对应Psi集合里面最大BC值的DM的调整方案集合
            
            
            %新的更新法则在Psi中
            %[q, a]=size(Omega); %求Omega里面DM的个数a
            
            
            D=DD;%DD是新赋值的初始评价信息
            for j=1:b
                B=[B BC(Psi(j))]; %存储Psi集合中DMs的BC值
                BB=[BB Psi(j)];  %存储Psi集合中的DM的标签
            end
            [~,E_q]=sort(B,'descend');  %对BC决策者按照冲突度进行从大到小排序
            d_k=BB(E_q(1)); %找出BC中冲突值最大的决策者
            disp(['Psi中冲突值最大的DM的标签',num2str(d_k)]) %输出d_k
            for j=1:b
                BCC=[BCC,r*(D(:,Psi(j)))+u*(D(:,d_k))]; %跟自身集合内的求平均
            end
            BCC(:,E_q(1))=[]; %去除掉自身的
            BCC=[BCC, r*(CA)+u*(D(:,d_k))];%跟CA求平均
            for zz=1:z
                BCC=[BCC,r*(D(:,Upsilon(z)))+u*(D(:,d_k))]; %跟没有冲突的DM求平均
            end
            
            BCC=[BCC sum(BCC,2)/(b+z)]; %求得的BCC再求平均
            BCC=roundn(BCC(:,:), -1); %结果保留小数点后一位
            EE=[];%更新后的评价信息 与原本DM的评价信息的差值
            bpt=[];%存储更新后的CD值
            
            for jj=1:b+z+1
                
                %C(jj)=zeros(NUM_DMs,NUM_DMs);
                EE=[EE BCC(:,jj)-D(:,d_k)];%计算差值
                D(:,d_k)=BCC(:,jj);%更新D 下面是就算更新替换D中第d_k列的评价信息后的CD值
                for i=1:NUM_DMs
                    y=D(:,i);
                    D_=D;
                    D_(:,i)=0;                      %AMR Operator
                    beta=SR(D_,y);
                    SR_coeffcients(:,i)=beta;       %column
                end
                BETA=SR_coeffcients;
                SR_coeffcients(SR_coeffcients>0)=0;
                C=abs(SR_coeffcients); %用于存储每次替换后求得的C矩阵 便于下一步的计算 d_k 统一则替换C到C（rb）
                %CC=C(jj);
                bp= 2*sum(sum(C(:,:)~=0))/(NUM_DMs*(NUM_DMs-1)); % group conflict degree CD
                
                bpt=[bpt bp];
                
            end
            
            
            
            [~,rb]=sort(bpt);%将更新得到的CD从小到大排列
            disp(['Psi集合的长度b=',num2str(b)])
            disp(['Upsilon集合的长度z=',num2str(z)])
            disp(['被调整专家在Psi集合中的位置',num2str(E_q(1))])
            disp(['最优调整方案对应BCC的列数',num2str(rb(1))])
            disp(['Dk_after=',num2str(BCC(:,rb(1))')])
            disp(['CD_after=',num2str(bpt(rb(1)))])
            A=input('输入行为冲突集决策者意愿(YES输入3,NO输入4):');
            
            
            % 新的更新法则在OC中
            while (A==3|isempty(Psi)&&~isempty(Omega)) %需要在OC 调整结束后加一个end 目前加了两个while
                %接下来3行是在Psi集合里面的操作
                DD(:,d_k)=BCC(:,rb(1));  %接受D_after，往下走在Omega集合中进行调整
                
                %重新计算C矩阵
                D=DD;
                for i=1:NUM_DMs
                    y=D(:,i);
                    D_=D;
                    D_(:,i)=0;                      %AMR Operator
                    beta=SR(D_,y);
                    SR_coeffcients(:,i)=beta;       %column
                end
                BETA=SR_coeffcients;
                SR_coeffcients(SR_coeffcients>0)=0;
                C=abs(SR_coeffcients); %新求得C用于计算下一步中的CD
                
                %CD=bpt(rb(1));
                CD=2*sum(sum(C(:,:)~=0))/(NUM_DMs*(NUM_DMs-1)); % group conflict degree CD
                
                AC=sum(sum(C(:,:)))/NUM_DMs;
                V=30*AC/CD;
                
                T=2*t-1;
                %% 存放计算结果
                result(T).D=D;  %群决策矩阵
                result(T).C=C;  %冲突矩阵矩阵
                result(T).k=d_k; %被调整的决策者的标签
                result(T).CD=CD;  %群体冲突度
                result(T).AC=AC;
                result(T).CDD=CD;
                result(T).V=V;
                %result(T).d_k=BCC(:,rb(1));%记录更新方案信息
                % 新的更新法则在OC中
                
                [q, a]=size(Omega); %求Omega里面DM的个数a
                %[w,b]=size(Psi);%求Psi里面DM的个数b
                %[V, z]=size(Upsilon); %求Upsilon 里面DM的个数
                for ii=1:a
                    AAA=[AAA OC(Omega(ii))]; %Omega中的DM对应的OC值
                    AA=[AA Omega(ii)];   %Omega中DM对应的标签
                end
                
                [~,E_q]=sort(AAA,'descend');  %对Omega中的决策者按照冲突度进行从大到小排序
                d_k=AA(E_q(1)); %找出Omega中冲突值最大的决策者
                disp(['Omega中冲突值最大的DM的标签',num2str(d_k)]) %输出d_k
                
                for ii=1:a
                    OCC=[OCC,r*(D(:,Omega(ii)))+u*(D(:,d_k))];%与Omega集合中的其他DM的评价信息加权平均
                end
                
                OCC(:,E_q(1))=[];%去除掉自身的
                OCC=[OCC, r*(CA)+u*(D(:,d_k))];%跟CA求平均
                for zz=1:z
                    OCC=[OCC,r*(D(:,Upsilon(z)))+u*(D(:,d_k))]; %跟没有冲突的DM求平均
                end
                
                OCC=[OCC sum(OCC,2)/(a+z)]; %OCC 集合自身求平均
                OCC=roundn(OCC(:,:), -1); %保留一位小数
                OEE=[];%OCC集合中的每一列与原始d_k列的差值
                opt=[]; %更新D（：，d_k）到OCC的每一列后得到的新的CD
                
                for ii=1:a+z+1
                    
                    OEE=[OEE OCC(:,ii)-D(:,d_k)];%计算差值
                    D(:,d_k)=OCC(:,ii);%更新D 的d_k列然后下面计算更新后的CD
                    for i=1:NUM_DMs
                        y=D(:,i);
                        D_=D;
                        D_(:,i)=0;                      %AMR Operator
                        beta=SR(D_,y);
                        SR_coeffcients(:,i)=beta;       %column
                    end
                    BETA=SR_coeffcients;
                    SR_coeffcients(SR_coeffcients>0)=0;
                    C=abs(SR_coeffcients);
                    op= 2*sum(sum(C(:,:)~=0))/(NUM_DMs*(NUM_DMs-1)); % group conflict degree CD
                    
                    
                    opt=[opt op];
                    
                end
                [~,rb]=sort(opt);%将更新得到的CD从小到大排列
                disp(['Omega集合的长度a=',num2str(a)])
                disp(['Upsilon集合的长度z=',num2str(z)])
                disp(['被调整专家在Omega集合中的位置',num2str(E_q(1))])
                disp(['最优调整方案对应OCC的列数',num2str(rb(1))])
                disp(['Dk_after=',num2str(OCC(:,rb(1))')])
                disp(['更新后的群体冲突度CD=',num2str(opt(rb(1)))])
                A=input('输入观点冲突集决策者意愿(YES输入1,NO输入2):');
                
                while(A==2) %Omega中DM拒绝接受调整
                    
                    Omega(find(Omega==d_k))=[];  %调整则决策者集合，并转入第14步 %怎么转 没看明白？
                    %重新计算C矩阵
                    D=DD;
                    
                    % 新的更新法则在OC中
                    [q, a]=size(Omega); %求Omega里面DM的个数a
                    %[w,b]=size(Psi);%求Psi里面DM的个数b
                    %[V, z]=size(Upsilon); %求Upsilon 里面DM的个数
                    AAA=[];
                    AA=[];
                    for ii=1:a
                        AAA=[AAA OC(Omega(ii))]; %Omega中的DM对应的OC值
                        AA=[AA Omega(ii)];   %Omega中DM对应的标签
                    end
                    
                    [~,E_q]=sort(AAA,'descend');  %对Omega中的决策者按照冲突度进行从大到小排序
                    d_k=AA(E_q(1)); %找出Omega中冲突值最大的决策者
                    disp(['Omega中冲突值最大的DM的标签',num2str(d_k)]) %输出d_k
                    
                    for ii=1:a
                        OCC=[OCC,r*(D(:,Omega(ii)))+u*(D(:,d_k))];%与Omega集合中的其他DM的评价信息加权平均
                    end
                    
                    OCC(:,E_q(1))=[];%去除掉自身的
                    OCC=[OCC, r*(CA)+u*(D(:,d_k))];%跟CA求平均
                    for zz=1:z
                        OCC=[OCC,r*(D(:,Upsilon(z)))+u*(D(:,d_k))]; %跟没有冲突的DM求平均
                    end
                    
                    OCC=[OCC sum(OCC,2)/(a+z)]; %OCC 集合自身求平均
                    OCC=roundn(OCC(:,:), -1); %保留一位小数
                    OEE=[];%OCC集合中的每一列与原始d_k列的差值
                    opt=[]; %更新D（：，d_k）到OCC的每一列后得到的新的CD
                    
                    for ii=1:a+z+1
                        
                        OEE=[OEE OCC(:,ii)-D(:,d_k)];%计算差值
                        D(:,d_k)=OCC(:,ii);%更新D 的d_k列然后下面计算更新后的CD
                        for i=1:NUM_DMs
                            y=D(:,i);
                            D_=D;
                            D_(:,i)=0;                      %AMR Operator
                            beta=SR(D_,y);
                            SR_coeffcients(:,i)=beta;       %column
                        end
                        BETA=SR_coeffcients;
                        SR_coeffcients(SR_coeffcients>0)=0;
                        C=abs(SR_coeffcients);
                        op= 2*sum(sum(C(:,:)~=0))/(NUM_DMs*(NUM_DMs-1)); % group conflict degree CD
                        
                        
                        opt=[opt op];
                        
                    end
                    [~,rb]=sort(opt);%将更新得到的CD从小到大排列
                    disp(['Omega集合的长度a=',num2str(a)])
                    disp(['Upsilon集合的长度z=',num2str(z)])
                    disp(['被调整专家在Omega集合中的位置',num2str(E_q(1))])
                    disp(['最优调整方案对应OCC的列数',num2str(rb(1))])
                    disp(['Dk_after=',num2str(OCC(:,rb(1))')])
                    disp(['更新后的群体冲突度CD=',num2str(opt(rb(1)))])
                    A=input('输入观点冲突集决策者意愿(YES输入1,NO输入2):');
                    
                    
                end
                
                while(A==1)
                    DD(:,d_k)=OCC(:,rb(1));  %接受D_after，并转至步骤5
                    E=[Omega Psi];
                    %重新计算C矩阵
                    D=DD;
                    for i=1:NUM_DMs
                        y=D(:,i);
                        D_=D;
                        D_(:,i)=0;                      %AMR Operator
                        beta=SR(D_,y);
                        SR_coeffcients(:,i)=beta;       %column
                    end
                    BETA=SR_coeffcients;
                    SR_coeffcients(SR_coeffcients>0)=0;
                    C=abs(SR_coeffcients); %新求得C用于计算下一步中的CD
                    CD=2*sum(sum(C(:,:)~=0))/(NUM_DMs*(NUM_DMs-1)); % group conflict degree CD
                    %CD=opt(rb(1));
                    AC=sum(sum(C(:,:)))/NUM_DMs;
                    V=30*AC/CD;
                    
                    %% 存放计算结果
                    TT=2*t;
                    result(TT).D=D;  %群决策矩阵
                    result(TT).C=C;  %冲突矩阵矩阵
                    result(TT).k=d_k; %被调整的决策者的标签
                    result(TT).CD=CD;  %群体冲突度
                    %result(TT).d_k=OCC(:,rb(1));%记录更新的评价信息
                    result(TT).AC=AC;
                    result(TT).CDD=CD;
                    result(TT).V=V;
                    
                    %D=DD;%重新分类
                    break
                    %Operation=input('本次循环结束是否继续(YES输入5,NO输入6):');
                end %while=1 end
                
            end%对应Omega调整的while A3
            
            %while =4
            while (A==4) %Psi中d_k拒绝调整方案
                
                Psi(find(Psi==d_k))=[];  %调整则Psi决策者集合，并转入第14步 %怎么转 --通过while循环语句转
                [q, a]=size(Omega); %求Omega里面DM的个数a
                [w,b]=size(Psi);%求Psi里面DM的个数b
                [V, z]=size(Upsilon); %求Upsilon 里面DM的个数
                B=[];
                BB=[];
                D=DD;%DD是新赋值的初始评价信息
                for j=1:b
                    B=[B BC(Psi(j))]; %存储Psi集合中DMs的BC值
                    BB=[BB Psi(j)];  %存储Psi集合中的DM的标签
                end
                [~,E_q]=sort(B,'descend');  %对BC决策者按照冲突度进行从大到小排序
                d_k=BB(E_q(1)); %找出BC中冲突值最大的决策者
                disp(['Psi中冲突值最大的DM的标签',num2str(d_k)]) %输出d_k
                
                for j=1:b
                    BCC=[BCC,r*(D(:,Psi(j)))+u*(D(:,d_k))]; %跟自身集合内的求平均
                end
                BCC(:,E_q(1))=[]; %去除掉自身的 %weile shiyan
                BCC=[BCC, r*(CA)+u*(D(:,d_k))];%跟CA求平均
                for zz=1:z
                    BCC=[BCC,r*(D(:,Upsilon(z)))+u*(D(:,d_k))]; %跟没有冲突的DM求平均
                end
                
                BCC=[BCC sum(BCC,2)/(b+z)]; %求得的BCC再求平均
                BCC=roundn(BCC(:,:), -1); %结果保留小数点后一位
                EE=[];%更新后的评价信息 与原本DM的评价信息的差值
                bpt=[];%存储更新后的CD值
                
                for jj=1:b+z+1
                    
                    %C(jj)=zeros(NUM_DMs,NUM_DMs);
                    EE=[EE BCC(:,jj)-D(:,d_k)];%计算差值
                    D(:,d_k)=BCC(:,jj);%更新D 下面是就算更新替换D中第d_k列的评价信息后的CD值
                    for i=1:NUM_DMs
                        y=D(:,i);
                        D_=D;
                        D_(:,i)=0;                      %AMR Operator
                        beta=SR(D_,y);
                        SR_coeffcients(:,i)=beta;       %column
                    end
                    BETA=SR_coeffcients;
                    SR_coeffcients(SR_coeffcients>0)=0;
                    C=abs(SR_coeffcients); %用于存储每次替换后求得的C矩阵 便于下一步的计算 d_k 统一则替换C到C（rb）
                    %CC=C(jj);
                    bp= 2*sum(sum(C(:,:)~=0))/(NUM_DMs*(NUM_DMs-1)); % group conflict degree CD
                    
                    bpt=[bpt bp];
                    
                end
                [~,rb]=sort(bpt);%将更新得到的CD从小到大排列
                disp(['Psi集合的长度b=',num2str(b)])
                disp(['Upsilon集合的长度z=',num2str(z)])
                disp(['被调整专家在Psi集合中的位置',num2str(E_q(1))])
                disp(['最优调整方案对应BCC的列数',num2str(rb(1))])
                disp(['Dk_after=',num2str(BCC(:,rb(1))')])
                disp(['CD_after=',num2str(bpt(rb(1)))])
                A=input('输入行为冲突集决策者意愿(YES输入3,NO输入4):');
                
                
                while (A==3|isempty(Psi)&&~isempty(Omega)) %需要在OC 调整结束后加一个end 目前加了两个while
                    %接下来3行是在Psi集合里面的操作
                    DD(:,d_k)=BCC(:,rb(1));  %接受D_after，往下走在Omega集合中进行调整
                    
                    %重新计算C矩阵
                    D=DD;
                    for i=1:NUM_DMs
                        y=D(:,i);
                        D_=D;
                        D_(:,i)=0;                      %AMR Operator
                        beta=SR(D_,y);
                        SR_coeffcients(:,i)=beta;       %column
                    end
                    BETA=SR_coeffcients;
                    SR_coeffcients(SR_coeffcients>0)=0;
                    C=abs(SR_coeffcients); %新求得C用于计算下一步中的CD
                    
                    CD=2*sum(sum(C(:,:)~=0))/(NUM_DMs*(NUM_DMs-1)); % group conflict degree CD
                    %CD=bpt(rb(1));
                    AC=sum(sum(C(:,:)))/NUM_DMs;
                    V=30*AC/CD;
                    
                    %% 存放计算结果
                    T=2*t-1;
                    result(T).D=D;  %群决策矩阵
                    result(T).C=C;  %冲突矩阵矩阵
                    result(T).dk=d_k; %被调整的决策者的标签
                    result(T).CD=CD;  %群体冲突度
                    result(T).AC=AC;
                    result(T).CDD=CD;
                    result(T).V=V;                    % 新的更新法则在OC中
                    [q, a]=size(Omega); %求Omega里面DM的个数a
                    %[w,b]=size(Psi);%求Psi里面DM的个数b
                    %[V, z]=size(Upsilon); %求Upsilon 里面DM的个数
                    for ii=1:a
                        AAA=[AAA OC(Omega(ii))]; %Omega中的DM对应的OC值
                        AA=[AA Omega(ii)];   %Omega中DM对应的标签
                    end
                    
                    [~,E_q]=sort(AAA,'descend');  %对Omega中的决策者按照冲突度进行从大到小排序
                    d_k=AA(E_q(1)); %找出Omega中冲突值最大的决策者
                    disp(['Omega中冲突值最大的DM的标签',num2str(d_k)]) %输出d_k
                    
                    for ii=1:a
                        OCC=[OCC,r*(D(:,Omega(ii)))+u*(D(:,d_k))];%与Omega集合中的其他DM的评价信息加权平均
                    end
                    
                    OCC(:,E_q(1))=[];%去除掉自身的
                    OCC=[OCC, r*(CA)+u*(D(:,d_k))];%跟CA求平均
                    for zz=1:z
                        OCC=[OCC,r*(D(:,Upsilon(z)))+u*(D(:,d_k))]; %跟没有冲突的DM求平均
                    end
                    
                    OCC=[OCC sum(OCC,2)/(a+z)]; %OCC 集合自身求平均
                    OCC=roundn(OCC(:,:), -1); %保留一位小数
                    OEE=[];%OCC集合中的每一列与原始d_k列的差值
                    opt=[]; %更新D（：，d_k）到OCC的每一列后得到的新的CD
                    
                    for ii=1:a+z+1
                        
                        OEE=[OEE OCC(:,ii)-D(:,d_k)];%计算差值
                        D(:,d_k)=OCC(:,ii);%更新D 的d_k列然后下面计算更新后的CD
                        for i=1:NUM_DMs
                            y=D(:,i);
                            D_=D;
                            D_(:,i)=0;                      %AMR Operator
                            beta=SR(D_,y);
                            SR_coeffcients(:,i)=beta;       %column
                        end
                        BETA=SR_coeffcients;
                        SR_coeffcients(SR_coeffcients>0)=0;
                        C=abs(SR_coeffcients);
                        op= 2*sum(sum(C(:,:)~=0))/(NUM_DMs*(NUM_DMs-1)); % group conflict degree CD
                        
                        
                        opt=[opt op];
                        
                    end
                    [~,rb]=sort(opt);%将更新得到的CD从小到大排列
                    disp(['Omega集合的长度a=',num2str(a)])
                    disp(['Upsilon集合的长度z=',num2str(z)])
                    disp(['被调整专家在Omega集合中的位置',num2str(E_q(1))])
                    disp(['最优调整方案对应OCC的列数',num2str(rb(1))])
                    disp(['Dk_after=',num2str(OCC(:,rb(1))')])
                    disp(['更新后的群体冲突度CD=',num2str(opt(rb(1)))])
                    A=input('输入观点冲突集决策者意愿(YES输入1,NO输入2):');
                    
                    while(A==2) %Omega中DM拒绝接受调整
                        Omega(find(Omega==d_k))=[];  %调整则决策者集合，并转入第14步 %怎么转 没看明白？
                        %重新计算C矩阵
                        D=DD;
                        
                        % 新的更新法则在OC中
                        [q, a]=size(Omega); %求Omega里面DM的个数a
                        %[w,b]=size(Psi);%求Psi里面DM的个数b
                        %[V, z]=size(Upsilon); %求Upsilon 里面DM的个数
                        AAA=[];
                        AA=[];
                        for ii=1:a
                            AAA=[AAA OC(Omega(ii))]; %Omega中的DM对应的OC值
                            AA=[AA Omega(ii)];   %Omega中DM对应的标签
                        end
                        
                        [~,E_q]=sort(AAA,'descend');  %对Omega中的决策者按照冲突度进行从大到小排序
                        d_k=AA(E_q(1)); %找出Omega中冲突值最大的决策者
                        disp(['Omega中冲突值最大的DM的标签',num2str(d_k)]) %输出d_k
                        
                        for ii=1:a
                            OCC=[OCC,r*(D(:,Omega(ii)))+u*(D(:,d_k))];%与Omega集合中的其他DM的评价信息加权平均
                        end
                        
                        OCC(:,E_q(1))=[];%去除掉自身的
                        OCC=[OCC, r*(CA)+u*(D(:,d_k))];%跟CA求平均
                        for zz=1:z
                            OCC=[OCC,r*(D(:,Upsilon(z)))+u*(D(:,d_k))]; %跟没有冲突的DM求平均
                        end
                        
                        OCC=[OCC sum(OCC,2)/(a+z)]; %OCC 集合自身求平均
                        OCC=roundn(OCC(:,:), -1); %保留一位小数
                        OEE=[];%OCC集合中的每一列与原始d_k列的差值
                        opt=[]; %更新D（：，d_k）到OCC的每一列后得到的新的CD
                        
                        for ii=1:a+z+1
                            
                            OEE=[OEE OCC(:,ii)-D(:,d_k)];%计算差值
                            D(:,d_k)=OCC(:,ii);%更新D 的d_k列然后下面计算更新后的CD
                            for i=1:NUM_DMs
                                y=D(:,i);
                                D_=D;
                                D_(:,i)=0;                      %AMR Operator
                                beta=SR(D_,y);
                                SR_coeffcients(:,i)=beta;       %column
                            end
                            BETA=SR_coeffcients;
                            SR_coeffcients(SR_coeffcients>0)=0;
                            C=abs(SR_coeffcients);
                            op= 2*sum(sum(C(:,:)~=0))/(NUM_DMs*(NUM_DMs-1)); % group conflict degree CD
                            
                            
                            opt=[opt op];
                            
                        end
                        [~,rb]=sort(opt);%将更新得到的CD从小到大排列
                        disp(['Omega集合的长度a=',num2str(a)])
                        disp(['Upsilon集合的长度z=',num2str(z)])
                        disp(['被调整专家在Omega集合中的位置',num2str(E_q(1))])
                        disp(['最优调整方案对应OCC的列数',num2str(rb(1))])
                        disp(['Dk_after=',num2str(OCC(:,rb(1))')])
                        disp(['更新后的群体冲突度CD=',num2str(opt(rb(1)))])
                        A=input('输入观点冲突集决策者意愿(YES输入1,NO输入2):');
                        
                        
                    end
                    
                    while(A==1)
                        DD(:,d_k)=OCC(:,rb(1));  %接受D_after，并转至步骤5
                        E=[Omega Psi];
                        %重新计算C矩阵
                        D=DD;
                        for i=1:NUM_DMs
                            y=D(:,i);
                            D_=D;
                            D_(:,i)=0;                      %AMR Operator
                            beta=SR(D_,y);
                            SR_coeffcients(:,i)=beta;       %column
                        end
                        BETA=SR_coeffcients;
                        SR_coeffcients(SR_coeffcients>0)=0;
                        C=abs(SR_coeffcients); %新求得C用于计算下一步中的CD
                        
                        %CD=opt(rb(1));
                        CD=2*sum(sum(C(:,:)~=0))/(NUM_DMs*(NUM_DMs-1)); % group conflict degree CD
                        AC=sum(sum(C(:,:)))/NUM_DMs;
                        V=30*AC/CD;
                        
                        %% 存放计算结果
                        TT=2*t;
                        result(TT).D=D;  %群决策矩阵
                        result(TT).C=C;  %冲突矩阵矩阵
                        result(TT).dk=d_k; %被调整的决策者的标签
                        result(TT).CD=CD;  %群体冲突度
                        result(TT).AC=AC;
                        result(TT).CDD=CD;
                        result(TT).V=V;                    %D=DD;%重新分类
                        break
                        %Operation=input('本次循环结束是否继续(YES输入5,NO输入6):');
                    end %while=1 end
                    
                end%对应Omega调整的while A3
                
            end%对应Omega调整的while A=4
            
            
        end%这个对应if t<=Tmax 进入一轮CRP的调整
        
    end
    % disp('FAILED')
    Operation=input('本次循环结束是否继续(YES输入5,NO输入6):');
    
end
%Operation=input('本次循环结束是否继续(YES输入5,NO输入6):');

























