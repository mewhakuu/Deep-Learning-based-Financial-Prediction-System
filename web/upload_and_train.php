<?php
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    // 检查文件是否已上传
    if (isset($_FILES["dataset"]) && $_FILES["dataset"]["error"] == 0) {
        // 获取参数
        $factor_dim = $_POST["factor_dim"];
        $L = $_POST["L"];
        $S = $_POST["S"];
        $batch_size = $_POST["batch_size"];
        $epoch = $_POST["epoch"];

        // 保存上传的文件
        $target_dir = "/www/wwwroot/192.168.67.129/";
        $target_file = $target_dir . basename($_FILES["dataset"]["name"]);
        move_uploaded_file($_FILES["dataset"]["tmp_name"], $target_file);

        // 运行Python脚本
        $command = escapeshellcmd("python train_model.py --dataset $target_file --factor_dim $factor_dim --L $L --S $S --batch_size $batch_size --epoch $epoch");
        $output = shell_exec($command);

        // 输出结果
        echo $output;
    } else {
        echo "文件上传出错。";
    }
}
?>