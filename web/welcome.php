<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>模型训练与预测</title>
<style>
  @keyframes gradientBackground {
  0% {
    background-position: 0% 50%;
  }
  50% {
    background-position: 100% 50%;
  }
  100% {
    background-position: 0% 50%;
  }
}

body {
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
  background-size: 400% 400%;
  animation: gradientBackground 15s ease infinite;
  margin: 0;
  padding: 0;
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100vh;
}

.form-container {
  /* ...保持其他样式不变... */
  animation: fadeInZoom 0.5s ease forwards;
}

input[type="submit"]:hover {
  background-color: #4cae4c;
}

input[type="submit"]:active {
  animation: buttonClick 0.2s ease;
}

@keyframes fadeInZoom {
  from {
    transform: scale(0.5);
    opacity: 0;
  }
  to {
    transform: scale(1);
    opacity: 1;
  }
}

@keyframes buttonClick {
  0% {
    transform: translateY(0);
  }
  50% {
    transform: translateY(-5px);
  }
  100% {
    transform: translateY(0);
  }
}

/* ...保持其他样式不变... */  
  body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background-color: #f4f4f4;
    margin: 0;
    padding: 0;
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100vh;
  }
  h2 {
    color: #333;
  }
  .form-container {
    background: #fff;
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.2);
  }
  form {
    display: flex;
    flex-direction: column;
  }
  label {
    margin-top: 10px;
  }
  input[type="file"],
  input[type="number"],
  input[type="submit"] {
    padding: 10px;
    margin-top: 5px;
    border: 1px solid #ddd;
    border-radius: 4px;
    box-sizing: border-box;
  }
  input[type="submit"] {
    background-color: #5cb85c;
    color: white;
    cursor: pointer;
    border: none;
  }
  input[type="submit"]:hover {
    background-color: #4cae4c;
  }
</style>
</head>
<body>

<div class="form-container">
  <h2>上传数据集并设置训练参数</h2>

  <form action="upload_and_train.php" method="post" enctype="multipart/form-data">
    <label for="dataset">数据集文件:</label>
    <input type="file" id="dataset" name="dataset" required>

    <label for="factor_dim">factor_dim:</label>
    <input type="number" id="factor_dim" name="factor_dim" required>

    <label for="L">L值:</label>
    <input type="number" id="L" name="L" required>

    <label for="S">S值:</label>
    <input type="number" id="S" name="S" required>

    <label for="batch_size">batch_size:</label>
    <input type="number" id="batch_size" name="batch_size" required>

    <label for="epoch">epoch:</label>
    <input type="number" id="epoch" name="epoch" required>

    <input type="submit" value="上传并训练">
  </form>
</div>

</body>
</html>
