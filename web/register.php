<?php
$host = "192.168.67.129"; // 数据库地址
$dbUsername = "192_168_67_129"; // 数据库用户
$dbPassword = "2PNF53484TE2"; // 数据库用户密码
$dbname = "192_168_67_129"; // 数据库名

// 创建连接
$conn = new mysqli($host, $dbUsername, $dbPassword, $dbname);

// 检查连接
if ($conn->connect_error) {
    die("连接失败: " . $conn->connect_error);
}

// 从表单获取数据
$username = $_POST['username'];
$password = $_POST['password'];

// 防止SQL注入
$username = stripslashes($username);
$password = stripslashes($password);
$username = $conn->real_escape_string($username);
$password = $conn->real_escape_string($password);

// 检查用户名是否已存在
$checkUserSql = "SELECT * FROM users WHERE username='$username'";
$checkUserResult = $conn->query($checkUserSql);
if ($checkUserResult->num_rows > 0) {
    // 用户名已存在
    echo '<!DOCTYPE html>
                <html lang="zh">
                <head>
                <meta charset="UTF-8">
                <title>用户名已存在</title>
                <style>
                @keyframes gradientBackground {
                    0% {background-position: 0% 50%;}
                    50% {background-position: 100% 50%;}
                    100% {background-position: 0% 50%;}
                }
                body {
                    font-family: Arial, sans-serif;
                    background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
                    background-size: 400% 400%;
                    animation: gradientBackground 15s ease infinite;
                    height: 100vh;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                }
                .error-container {
                    text-align: center;
                    color: white;
                }
                </style>
                </head>
                <body>
                    <div class="success-container">
                        <h1>用户名已存在</h1>
                        <p>5秒后将自动跳转到登录界面。</p>
                    </div>
                    <script>
                    setTimeout(function(){
                        window.location.href = \'index.html\';
                    }, 5000);
                    </script>
                    </body>
                </html>';
} else {
    // 将数据插入数据库
    $sql = "INSERT INTO users (username, password) VALUES ('$username', '$password')";
    if ($conn->query($sql) === TRUE) {
        echo '<!DOCTYPE html>
            <html lang="zh">
            <head>
            <meta charset="UTF-8">
            <title>注册成功</title>
            <style>
            @keyframes gradientBackground {
                0% {background-position: 0% 50%;}
                50% {background-position: 100% 50%;}
                100% {background-position: 0% 50%;}
            }
            body {
                font-family: Arial, sans-serif;
                background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
                background-size: 400% 400%;
                animation: gradientBackground 15s ease infinite;
                height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
            }
            .error-container {
                text-align: center;
                color: white;
            }
            </style>
            </head>
            <body>
                <div class="success-container">
                    <h1>注册成功！</h1>
                    <p>5秒后将自动跳转到登录界面。</p>
                </div>
                <script>
                setTimeout(function(){
                    window.location.href = \'index.html\';
                }, 5000);
                </script>
                </body>
            </html>';
    } else {
        echo "Error: " . $sql . "<br>" . $conn->error;
    }
}

$conn->close();
?>
