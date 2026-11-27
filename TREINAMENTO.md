# Treinamento pelo painel MabiYama

1. Execute **Iniciar MabiYama.cmd** e mantenha a janela aberta. Na primeira execução, o atalho instala as dependências se necessário.
2. Abra http://127.0.0.1:8000 e entre com o login do painel.
3. Abra **Treinamento**, selecione **Telemetria**, **Máquinas**, **Operação**, **Falhas**, **Manutenção** e **Erros** e clique em **Enviar e treinar**.
4. Quando o treinamento terminar, use **Ver previsões** ou **Baixar submission.csv**. O servidor mantém a última submission gerada para a próxima abertura.

Pelo terminal, o comando equivalente, com as dependências instaladas, é `python servidor.py`. A instalação manual usa `python -m pip install -r requirements.txt`.

O treinamento fornecido usa rótulos até 29/09/2023 e gera previsões de 30/09/2023 a 30/12/2023. Os seis arquivos devem pertencer à mesma frota e conter dados suficientes para esse período. O `sample_submission.csv` incluído ordena as linhas apenas quando a frota enviada coincide com a do modelo.

O servidor Python atende apenas no próprio computador. Abrir `index.html` diretamente ou servir o HTML em uma hospedagem estática permite abrir o painel, mas o botão de treinamento depende do servidor Python.
