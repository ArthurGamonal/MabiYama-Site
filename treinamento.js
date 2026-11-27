(() => {
  const api = '/api/treinamento';
  const form = document.getElementById('treinoForm');
  const botao = document.getElementById('treinoEnviar');
  const status = document.getElementById('treinoStatus');
  const etapas = [...document.querySelectorAll('#treinoEtapas li')];
  const baixar = document.getElementById('treinoBaixar');
  const ver = document.getElementById('treinoVer');
  let versao = null;
  let mensagemResultado = '';
  const abertoComoArquivo = location.protocol === 'file:';

  async function requisitar(url, opcoes) {
    const resposta = await fetch(url, { cache:'no-store', ...opcoes });
    if (resposta.status === 404 && url.endsWith('/resultado')) return null;
    const dados = await resposta.json();
    if (!resposta.ok) throw new Error(dados.erro || 'Não foi possível acessar o servidor de treinamento.');
    return dados;
  }

  function estadoDaTela(mensagem, erro = false) {
    status.textContent = mensagem;
    status.classList.toggle('erro', erro);
  }

  async function carregarResultado() {
    const dados = await requisitar(api + '/resultado');
    if (!dados) return;
    baixar.hidden = false;
    if (versao === dados.id || localStorageSeguro('treinoIgnorado') === dados.id) return;
    aplicarTreino(dados);
    versao = dados.id;
    mensagemResultado = `Painel atualizado com a previsão de ${dados.maquinas.length} máquinas.`;
    ver.hidden = false;
    if (estado.aba === 'campo') campoAbrirDoLink();
    return dados;
  }

  async function acompanhar() {
    const { trabalho } = await requisitar(api);
    if (!trabalho) return;
    const emAndamento = ['recebendo', 'executando'].includes(trabalho.estado);
    botao.disabled = emAndamento;
    form.setAttribute('aria-busy', String(emAndamento));
    const i = trabalho.etapa;
    etapas.forEach((el, n) => {
      el.classList.toggle('concluida', n < i - 1 || trabalho.estado === 'concluido');
      el.classList.toggle('corrente', n === i - 1 && trabalho.estado !== 'concluido');
      if (n === i - 1 && emAndamento) el.setAttribute('aria-current', 'step');
      else el.removeAttribute('aria-current');
    });
    if (emAndamento) {
      estadoDaTela(i ? etapas[i - 1].textContent + '…' : 'Recebendo os CSVs…');
      setTimeout(() => acompanhar().catch(mostrarFalha), 1500);
    } else if (trabalho.estado === 'erro') {
      estadoDaTela(trabalho.erro || 'O treinamento falhou.', true);
    } else if (trabalho.estado === 'concluido') {
      await carregarResultado();
      estadoDaTela(idTreinoAtivo ? mensagemResultado : 'Treinamento concluído. A submission está disponível.');
    }
  }

  function mostrarFalha(erro) {
    botao.disabled = false;
    form.setAttribute('aria-busy', 'false');
    estadoDaTela(erro.message === 'Failed to fetch' ? 'Não foi possível conectar ao servidor local.' : erro.message, true);
  }

  function localStorageSeguro(chave) {
    try { return localStorage.getItem(chave); } catch (erro) { return null; }
  }

  form.addEventListener('submit', async evento => {
    evento.preventDefault();
    if (abertoComoArquivo || botao.disabled) return;
    botao.disabled = true;
    form.setAttribute('aria-busy', 'true');
    etapas.forEach(el => el.classList.remove('concluida', 'corrente'));
    estadoDaTela('Enviando os seis CSVs…');
    try {
      await requisitar(api, { method:'POST', body:new FormData(form), headers:{ 'X-MabiYama':'treinamento' } });
      await acompanhar();
    } catch (erro) {
      mostrarFalha(erro);
    }
  });
  ver.addEventListener('click', () => mostrarAba(visitasAmanha(estado.dia) ? 'simples' : 'maquinas'));

  if (abertoComoArquivo) {
    botao.disabled = true;
    estadoDaTela('Inicie o servidor Python e abra http://127.0.0.1:8000 para treinar.');
  } else {
    carregarResultado().then(dados => {
      if (dados) estadoDaTela(mensagemResultado);
      return acompanhar();
    }).catch(erro => {
      estadoDaTela('Servidor de treinamento indisponível. Abra o painel pelo servidor Python local.', true);
      botao.disabled = true;
      console.warn(erro);
    });
  }
})();
