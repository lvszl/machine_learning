function strimage(path, n)
  fidin = fopen(path); % 打开train-01-images.svm文件
  i = 1; % 初始化计数器

  apres = []; % 创建空数组用于存储数据

  while ~feof(fidin) % 当未到达文件末尾时循环
    tline = fgetl(fidin); % 从文件中读取一行数据
    apres{i} = tline; % 将数据存入数组
    i = i+1; % 计数器增加
  end

  a = char(apres(n)); % 获取数组中第 n 行数据

  lena = size(a); % 获取字符串长度
  lena = lena(2); % 获取字符串的第二维度长度

  xy = sscanf(a(4:lena), '%d:%d'); % 从字符串中提取数据

  lenxy = size(xy); % 获取提取的数据长度
  lenxy = lenxy(1); % 获取数据的第一维度长度

  grid = []; % 创建空数组用于存储像素数据
  grid(784) = 0; % 初始化数组大小

  for i=2:2:lenxy  % 从第二个数开始，每隔一个数取值
      if(xy(i)<=0) % 如果值小于等于0，跳出循环
          break
      end
    grid(xy(i-1)) = xy(i) * 100/255; % 计算像素值并存入数组
  end

  grid1 = reshape(grid,28,28); % 重塑数组为28x28的矩阵
  grid1 = fliplr(diag(ones(28,1)))*grid1; % 对矩阵进行反转和对角线操作
  grid1 = rot90(grid1,3); % 逆时针旋转矩阵

  image(grid1) % 显示图像
  hold on; % 保持图像在同一画布上
end
