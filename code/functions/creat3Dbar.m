function creat3Dbar(DATA)

hb=bar3(DATA);
shading interp
for j=1:length(hb)
    zdata=get(hb(j),'Zdata');
    set(hb(j),'Cdata',zdata)
end
end