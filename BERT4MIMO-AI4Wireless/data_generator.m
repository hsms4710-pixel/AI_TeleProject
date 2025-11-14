% Parameters / 参数
numCells = 10;              % Number of base stations / 基站数量
numUEs = 200;               % Number of user equipment (UE) per cell / 每个小区的用户数量
numSubcarriers = 64;        % Number of subcarriers / 子载波数量
massiveMIMONumAntennas = 64; % Number of antennas for Massive MIMO (base station) / 基站天线数（大规模 MIMO）
numReceiveAntennas = 4;     % Number of antennas at the user equipment / 用户端天线数
nrSampleRate = 30.72e6;     % Sample rate for 5G NR / 5G NR 采样率
snrNR = 25;                 % Signal-to-Noise Ratio for 5G NR (in dB) / 信噪比（dB）
speedHigh = 120;            % User speed for high mobility (in km/h) / 高速场景用户速度（km/h）
fc = 3.5e9;                 % Carrier frequency (3.5 GHz for 5G NR) / 载波频率（Hz）

% Calculate Doppler shift for high-speed mobility / 计算高速场景的多普勒频移
c = physconst('lightspeed'); % Speed of light in m/s / 光速（m/s）
maxDopplerShiftHigh = (speedHigh * 1000 / 3600) / c * fc; % Doppler shift in Hz / 最大多普勒频移（Hz）

% Initialize data storage / 初始化数据存储结构
multi_cell_csi = cell(numCells, numUEs); % CSI storage for all cells and UEs / 存放所有小区和 UE 的 CSI

% Function to add AWGN to the CSI / 向 CSI 添加高斯白噪声的函数句柄
addNoise = @(csi, snr) awgn(csi, snr, 'measured'); % 使用 MATLAB awgn，按测量信号功率添加噪声

fprintf('Processing progress:\n'); % 打印进度标题
totalIterations = numCells * numUEs; % 总迭代次数（用于进度显示）
iterationCounter = 0; % Counter to track progress / 迭代计数器
startTime = datetime('now'); % Start time / 开始时间，用于估算速率

%% Massive MIMO Scenarios / 大规模 MIMO 场景循环
for cellIdx = 1:numCells
    for ueIdx = 1:numUEs
        % Update progress / 更新进度计数并打印
        iterationCounter = iterationCounter + 1;
        elapsedTime = seconds(datetime('now') - startTime); % 已用秒数
        itemsPerSecond = iterationCounter / max(elapsedTime, 1e-6); % 防止除以零

        % Display progress with timestamp and items per second
        % 显示当前时间戳、百分比、已处理数 / 总数、处理速率（项/秒）
        fprintf('\r[%s] Progress: %.2f%% (%d/%d) | %.2f items/sec', ...
            datestr(datetime('now')), ...
            (iterationCounter / totalIterations) * 100, ...
            iterationCounter, ...
            totalIterations, ...
            itemsPerSecond);

        % Define Massive MIMO channel model for different scenarios
        % 为不同场景定义 nrTDLChannel 信道模型

        % Scenario 1: Stationary UE with standard TDL-A
        % 场景1：静止 UE，TDL-A 延迟型（短延迟扩散）
        channel_stationary = nrTDLChannel(...
            'DelayProfile', 'TDL-A', ...
            'DelaySpread', 100e-9, ...
            'NumTransmitAntennas', massiveMIMONumAntennas, ...
            'NumReceiveAntennas', numReceiveAntennas, ...
            'SampleRate', nrSampleRate);

        % Scenario 2: High-speed UE (e.g., car moving at high speed)
        % 场景2：高速移动 UE（TDL-C），包含多普勒效应
        channel_high_speed = nrTDLChannel(...
            'DelayProfile', 'TDL-C', ...
            'DelaySpread', 300e-9, ...
            'MaximumDopplerShift', maxDopplerShiftHigh, ... % Simulate high-speed mobility / 模拟高速度
            'NumTransmitAntennas', massiveMIMONumAntennas, ...
            'NumReceiveAntennas', numReceiveAntennas, ...
            'SampleRate', nrSampleRate);

        % Scenario 3: Urban Macrocell with longer delay spread
        % 场景3：城市宏小区，较大延迟扩散（TDL-D）
        channel_urban_macro = nrTDLChannel(...
            'DelayProfile', 'TDL-D', ...
            'DelaySpread', 500e-9, ...
            'NumTransmitAntennas', massiveMIMONumAntennas, ...
            'NumReceiveAntennas', numReceiveAntennas, ...
            'SampleRate', nrSampleRate);

        % Generate CSI for each subcarrier in each scenario
        % 对每个场景的每个子载波生成 CSI
        scenarios = {channel_stationary, channel_high_speed, channel_urban_macro};
        csi_scenarios = cell(1, numel(scenarios)); % Store CSI for all scenarios / 存放每个场景的 CSI

        for scenarioIdx = 1:numel(scenarios)
            channel = scenarios{scenarioIdx};
            % Initialize CSI matrix: Subcarrier x Tx x Rx
            % 初始化 CSI 张量，维度为 子载波 × 发射天线 × 接收天线
            csi = zeros(numSubcarriers, massiveMIMONumAntennas, numReceiveAntennas);

            % Use identity matrix as dummy signal to probe each Tx antenna separately
            % 使用复单位矩阵作为输入，模拟逐天线激励以获取 MIMO path gains
            dummySignal = complex(eye(massiveMIMONumAntennas)); % Identity matrix, complex

            % Generate path gains for each subcarrier
            % 对每个子载波调用信道模型，获得 pathGains（Tx x Rx）
            for subcarrierIdx = 1:numSubcarriers
                [pathGains, ~] = channel(dummySignal); % 输出维度：[numTransmitAntennas x numReceiveAntennas]
                reshapedPathGains = pathGains; % pathGains 已为 Tx x Rx，无需重塑
                csi(subcarrierIdx, :, :) = reshapedPathGains; % 填充到 CSI 矩阵
            end

            % Add noise to the channel / 向 CSI 添加噪声以模拟接收端噪声
            noisyCsi = addNoise(csi, snrNR);

            % Store noisy CSI for the scenario / 存储带噪声的 CSI
            csi_scenarios{scenarioIdx} = noisyCsi;
        end

        % Store all scenarios for the given cell and UE
        % 将该小区、该 UE 的所有场景 CSI 存入 multi_cell_csi 单元格数组
        multi_cell_csi{cellIdx, ueIdx} = csi_scenarios;
    end
end

%% Save All Data to a Single MAT File / 将所有数据保存为单个 MAT 文件
% 保存变量 multi_cell_csi 到指定路径
save('foundation_model_data/csi_data_massive_mimo.mat', ...
    'multi_cell_csi');