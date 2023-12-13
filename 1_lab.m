% Classification using perceptron
% Reading apple images
A1=imread('apple_04.jpg');
A2=imread('apple_05.jpg');
A3=imread('apple_06.jpg');
A4=imread('apple_07.jpg');
A5=imread('apple_11.jpg');
A6=imread('apple_12.jpg');
A7=imread('apple_13.jpg');
A8=imread('apple_17.jpg');
A9=imread('apple_19.jpg');
% Reading pears images
P1=imread('pear_01.jpg');
P2=imread('pear_02.jpg');
P3=imread('pear_03.jpg');
P4=imread('pear_09.jpg');
% Calculate for each image, colour and roundness
% For Apples
% 1st apple image(A1)
hsv_value_A1=spalva_color(A1); %color
metric_A1=apvalumas_roundness(A1); %roundness
% 2nd apple image(A2)
hsv_value_A2=spalva_color(A2); %color
metric_A2=apvalumas_roundness(A2); %roundness
% 3rd apple image(A3)
hsv_value_A3=spalva_color(A3); %color
metric_A3=apvalumas_roundness(A3); %roundness
% 4th apple image(A4)
hsv_value_A4=spalva_color(A4); %color
metric_A4=apvalumas_roundness(A4); %roundness
% 5th apple image(A5)hsv_value_A5=spalva_color(A5); %color
metric_A5=apvalumas_roundness(A5); %roundness
% 6th apple image(A6)
hsv_value_A6=spalva_color(A6); %color
metric_A6=apvalumas_roundness(A6); %roundness
% 7th apple image(A7)
hsv_value_A7=spalva_color(A7); %color
metric_A7=apvalumas_roundness(A7); %roundness
% 8th apple image(A8)
hsv_value_A8=spalva_color(A8); %color
metric_A8=apvalumas_roundness(A8); %roundness
% 9th apple image(A9)
hsv_value_A9=spalva_color(A9); %color
metric_A9=apvalumas_roundness(A9); %roundness
%For Pears
%1st pear image(P1)
hsv_value_P1=spalva_color(P1); %color
metric_P1=apvalumas_roundness(P1); %roundness
%2nd pear image(P2)
hsv_value_P2=spalva_color(P2); %color
metric_P2=apvalumas_roundness(P2); %roundness
%3rd pear image(P3)
hsv_value_P3=spalva_color(P3); %color
metric_P3=apvalumas_roundness(P3); %roundness
%2nd pear image(P4)
hsv_value_P4=spalva_color(P4); %color
metric_P4=apvalumas_roundness(P4); %roundness
%selecting features(color, roundness, 3 apples and 2 pears)
%A1,A2,A3,P1,P2
%building matrix 2x5
x1=[hsv_value_A1 hsv_value_A2 hsv_value_A3 hsv_value_P1 hsv_value_P2]; %3 obuoliu spalvos ir 2 kriausiu spalvos
x2=[metric_A1 metric_A2 metric_A3 metric_P1 metric_P2]; %3 obuoliu apvalumas ir 2 kriausiu apvalumas
% estimated features are stored in matrix P:
P=[x1;x2];
T=[1;1;1;-1;-1]; 
%% train single perceptron with two inputs and one output
% generate random initial values of w1, w2 and b
w1 = randn(1); %parenku atsitiktinius svorius
w2 = randn(1);
b = randn(1);%bias
% calculate weighted sum with randomly generated parameters
v = 0; % write your code here
% calculate current output of the perceptron
for i=1:5
   v(i)=P(1,i)*w1+(P(2,i)*w2)+b
if v(i) > 0
    %y yra isejimas, priskiria vienai klasei (1) arba kitai (-1)
y = 1; %klasifikuoja vienai klasei
   
else
y = -1;%klasifikuoja kitai klasei
end
e(i)=T(i)-y % skaiciuoju klaida iteracijoj (klaida bus nulis jei norimas outputas ir gautas outputas yra vienodi, aka ju skirtumas lygus 0)
end
e=sum(abs(e))
disp(v)
%      0<eta<1
eta=0.01; %pasirenku mokymo zingsni(kaip greitai algoritmas mokosi)
%kokiu zingsniu atnaujina svoriu reiksmes
disp("----------------------------------")
count=0;  %skaitiklis, ciklo iteraciju skaiciavimui
while e ~= 0
   %ciklas sukasi tol kol klaida patampa nulis
    for i=1:5
       
        v=P(1,i)*w1+(P(2,i)*w2)+b; %skaiciuoju atsaka
       
       if v>0 %skaiciuoju outputa
             y=1;
        else
           y=-1;
       end
       %kuo mazesnis mokymo zingsnis (0.001) tuo mazesniu zingsniu
       %algoritmas atnaujina svoriu reiksmes, mazesnis zingnis =daugiau
       %mokymo iteraciju
         e(i)=T(i)-y % skaiciuoju klaida
        w1 =w1+eta*e(i)*P(1,i);% atsinaujinu svorius
        w2 =w2+eta*e(i)*P(2,i);
        b  =b+eta*e(i); %atsinaujinu bias
       
    end
     
 e = sum(abs(e)) % sumuoju kiekvienos iteracijos klaidas
 %jei e(klaida) nera nulis, ciklas suksis is naujo
 count=count+1 %skaiciuoju iteraciju skaiciu
 %e=0; %tikrinu, kaip prasisuka pirmas ciklas
end
disp(count) %atvaizduoju kiek ciklu reikejo algoritmui suklasifikuoti obuolius
%ir kriauses teisingai
