"""
Interface de Chat Interativo — EcoMental
=========================================
Loop principal que conecta o usuário ao CentralAgent.

Comandos:
  /help    — ajuda
  /stats   — estatísticas do agente
  /memory  — lista os itens da memória episódica
  /reset   — reseta a conversa (mantém memória e pesos)
  /save    — salva estado manualmente
  /clear   — limpa a tela
  /self    — inicia diálogo interno (IA fala consigo mesma)
  /exit    — sai
"""

import os
import sys
from typing import Optional

try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    HAS_COLOR = True
except ImportError:
    HAS_COLOR = False

    class Fore:
        CYAN = GREEN = YELLOW = RED = BLUE = MAGENTA = WHITE = ""

    class Style:
        BRIGHT = RESET_ALL = DIM = ""


from config import Config
from brain.agent import CentralAgent


SELF_NAME = "Deorita"
SELF_MARKER = "deorita:"
SELF_DIALOGUE_TAG = "[AUTO_DIALOGO_INTERNO]"


def _get_reddit_assistant(agent: CentralAgent):
    if not hasattr(agent, "reddit_assistant"):
        from reddit_assistant import DeoritaRedditAssistant

        agent.reddit_assistant = DeoritaRedditAssistant(agent, agent.cfg)
    return agent.reddit_assistant

HELP_TEXT = """
Comandos disponíveis:
  /help    — mostra esta ajuda
  /stats   — estatísticas do agente (steps, memória, recompensa)
  /memory  — lista os 10 itens de maior prioridade na memória
  /search <tema>         — pesquisa na web e traz um resumo legível
  /auto_search           — liga/desliga pesquisa autônoma periódica
  /explore [n]           — executa n pesquisas autônomas em sequência
  /web_chat <url> <msg>  — conversa com outra IA em página web
    /web_click_chat <msg>  — usa templates de imagem para clicar/colar/enviar/copiar
    /web_click_cycle [n] [tema] — ciclo entre IA local e IA externa (click bot)
  /reddit                — ajuda do modo Reddit assistido
  /reset   — reseta a conversa (mantém memória e pesos aprendidos)
  /save    — salva estado manualmente
  /clear   — limpa a tela
  /self [tema] [turnos] — inicia diálogo interno (IA fala consigo mesma)
                          Ex: /self curiosidade 5
  /feedback             — exibe histórico de autoavaliações da IA
  /exit    — encerra o programa
"""

# (sem prompts fixos — a IA decide o caminho do diálogo por conta própria)


def _clear():
    os.system("cls" if os.name == "nt" else "clear")


def print_header():
    sep = "=" * 60
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{sep}")
    print(f"  EcoMental — A IA que pensa, lembra e decide o que aprender")
    print(f"  Memória Episódica · Raciocínio Interno · Autoaprendizado")
    print(f"{sep}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}  /help para ver comandos{Style.RESET_ALL}\n")


def print_metrics(metrics: dict):
    r = metrics.get("r_t", 0.0)
    g = metrics.get("g_t", 0.0)
    mem = metrics.get("memory_size", 0)
    stp = metrics.get("step", 0)

    # Estado interno e estabilidade
    v_t = metrics.get("V_t", 0.0)
    surprise = metrics.get("surprise", 0.0)
    unc = metrics.get("self_uncertainty", 0.0)
    k_used = metrics.get("k_used", "-")
    context_k = metrics.get("context_k", "-")
    p_norm = metrics.get("personality_norm", 0.0)

    # Qualidade do aprendizado
    l_world = metrics.get("L_world", 0.0)
    l_self = metrics.get("L_self", 0.0)
    l_policy = metrics.get("L_policy", 0.0)
    l_critic = metrics.get("L_critic", 0.0)
    l_replay = metrics.get("L_replay", 0.0)

    # Feedback cíclico e recompensa acumulada
    fb_mod = metrics.get("feedback_mod", 1.0)
    fb_trend = metrics.get("feedback_trend", "estável")
    total_reward = metrics.get("total_reward", 0.0)
    avg_reward = total_reward / max(stp, 1)

    reward_color = Fore.GREEN if r > 0 else Fore.YELLOW
    if r < -0.2:
        reward_color = Fore.RED

    trend_tag = "="
    if str(fb_trend).lower().startswith("melhor"):
        trend_tag = "^"
    elif str(fb_trend).lower().startswith("pior"):
        trend_tag = "v"

    print(
        f"\n{Style.DIM}{reward_color}"
        f"[r={r:.3f}  g={g:.3f}  V={v_t:+.3f}  sur={surprise:.3f}  unc={unc:.3f}  mem={mem}  step={stp}]"
        f"{Style.RESET_ALL}"
    )
    print(
        f"{Style.DIM}"
        f"[Lw={l_world:.3f}  Ls={l_self:.3f}  Lp={l_policy:.3f}  Lc={l_critic:.3f}  Lr={l_replay:.3f}  k={k_used}  ctx={context_k}]"
        f"{Style.RESET_ALL}"
    )
    print(
        f"{Style.DIM}"
        f"[fb={fb_mod:.2f}({trend_tag})  p_norm={p_norm:.3f}  Rtot={total_reward:.2f}  Ravg={avg_reward:.3f}]"
        f"{Style.RESET_ALL}"
    )

    if metrics.get("self_talk_detected", False):
        reason = metrics.get("self_talk_reason", "auto-conversa detectada")
        print(
            f"{Fore.YELLOW}{Style.DIM}"
            f"[auto-conversa] {reason}"
            f"{Style.RESET_ALL}"
        )


def _has_deorita_marker(text: str) -> bool:
    raw = (text or "").strip().lower()
    return raw.startswith(SELF_MARKER) or raw.startswith("[deorita]")


def handle_command(cmd: str, agent: CentralAgent) -> bool:
    """
    Processa comandos especiais.
    Retorna False se o usuário quer sair.
    """
    raw_cmd = cmd.strip()
    cmd = raw_cmd.lower()

    if cmd == "/help":
        print(f"{Fore.YELLOW}{HELP_TEXT}{Style.RESET_ALL}")

    elif cmd == "/stats":
        stats = agent.get_stats()
        print(f"\n{Fore.CYAN}Estatísticas:{Style.RESET_ALL}")
        for k, v in stats.items():
            print(f"  {k}: {v}")

    elif cmd == "/memory":
        mem = agent.memory
        print(f"\n{Fore.CYAN}Memória Episódica ({len(mem)} itens):{Style.RESET_ALL}")
        choices = agent.memory.get_agent_choice_summary()
        print(f"{Style.DIM}  🔒 guardando: {choices['guardar']}  💨 soltando: {choices['soltar']}  · neutro: {choices['neutro']}{Style.RESET_ALL}")
        if len(mem) == 0:
            print("  (vazia)")
        else:
            items = sorted(
                [
                    (txt, prio, pin, ov)
                    for txt, prio, pin, ov in zip(
                        mem.texts, mem.priorities, mem.is_pinned, mem.agent_decay_override
                    )
                ],
                key=lambda x: x[1],
                reverse=True,
            )[:10]
            for i, (text, prio, pin, ov) in enumerate(items, 1):
                if pin:
                    tag = f"{Fore.YELLOW}[📌 perfil]{Style.RESET_ALL} "
                elif ov is not None and ov <= 0.1:
                    tag = f"{Fore.GREEN}[🔒 guardar]{Style.RESET_ALL} "
                elif ov is not None and ov >= 2.0:
                    tag = f"{Fore.RED}[💨 soltar]{Style.RESET_ALL} "
                else:
                    tag = ""
                print(f"  {i}. [{prio:.3f}] {tag}{text[:80]}")

    elif cmd.startswith("/search"):
        query = raw_cmd[len("/search"):].strip()
        if not query:
            print(f"{Fore.YELLOW}  Uso: /search <o que pesquisar>{Style.RESET_ALL}")
        else:
            print(f"{Fore.CYAN}  Pesquisando:{Style.RESET_ALL} {query}")
            result = agent.search_web(query)
            print(f"\n{Style.DIM}{result}{Style.RESET_ALL}")

    elif cmd == "/auto_search":
        agent.autonomous_search = not bool(getattr(agent, "autonomous_search", False))
        status = "ligada" if agent.autonomous_search else "desligada"
        print(f"{Fore.GREEN}  Pesquisa autônoma {status}.{Style.RESET_ALL}")

    elif cmd.startswith("/explore"):
        parts = raw_cmd.split(maxsplit=1)
        try:
            num_queries = int(parts[1]) if len(parts) > 1 else 3
        except ValueError:
            num_queries = 3
        _run_autonomous_research(agent, num_queries=num_queries)

    elif cmd.startswith("/web_chat"):
        # Usa primeiro espaço como separador entre URL e mensagem
        parts = raw_cmd[len("/web_chat"):].strip().split(maxsplit=1)
        if len(parts) < 2:
            print(f"{Fore.YELLOW}  Uso: /web_chat <url_da_ia> <sua_mensagem>{Style.RESET_ALL}")
            print(f"{Style.DIM}  Ex: /web_chat https://chatgpt.com oi como voce funciona?{Style.RESET_ALL}")
        else:
            url = parts[0]
            message = parts[1]
            print(f"{Fore.CYAN}  Conectando a {url[:60]}...{Style.RESET_ALL}")
            print(f"{Style.DIM}  Enviando: {message[:80]}...{Style.RESET_ALL}\n")
            result = agent.chat_with_external_ai(url, message)
            print(f"{Fore.GREEN}  IA Externa respondeu:{Style.RESET_ALL}")
            print(f"{Style.DIM}{result}{Style.RESET_ALL}\n")

    elif cmd.startswith("/web_click_chat"):
        message = raw_cmd[len("/web_click_chat"):].strip()
        if not message:
            print(f"{Fore.YELLOW}  Uso: /web_click_chat <sua_mensagem>{Style.RESET_ALL}")
            print(f"{Style.DIM}  Templates esperados em templates/web_click/: dialog.png, send.png, copy.png{Style.RESET_ALL}")
        else:
            print(f"{Fore.CYAN}  Bot mecânico por imagem em execução...{Style.RESET_ALL}")
            print(f"{Style.DIM}  Enviando: {message[:80]}...{Style.RESET_ALL}\n")
            result = agent.chat_with_external_ai_clickbot(message)
            print(f"{Fore.GREEN}  IA Externa respondeu:{Style.RESET_ALL}")
            print(f"{Style.DIM}{result}{Style.RESET_ALL}\n")

    elif cmd.startswith("/web_click_cycle"):
        _run_web_click_cycle(raw_cmd, agent)

    elif cmd == "/reddit":
        print(
            f"{Fore.BLUE}Modo Reddit assistido:{Style.RESET_ALL}\n"
            "  /reddit scan               — detecta posts visíveis na tela\n"
            "  /reddit inspect <n>        — lê e analisa o post n\n"
            "  /reddit draft <n>          — gera rascunho de comentário para o post n\n"
            "  /reddit open <n>           — abre o post n no navegador\n"
            "  /reddit like <n>           — envia upvote explícito no post n\n"
            "  /reddit comment <n>        — envia o rascunho atual no post n\n"
            "  /reddit status             — mostra estado do assistente\n"
            f"{Style.DIM}Likes e comentários só acontecem por comando explícito seu.{Style.RESET_ALL}"
        )

    elif cmd.startswith("/reddit "):
        _handle_reddit_command(raw_cmd, agent)

    elif cmd == "/reset":
        agent.reset_conversation()
        print(f"{Fore.GREEN}  Conversa resetada (memória e pesos preservados){Style.RESET_ALL}")

    elif cmd == "/save":
        agent.save()
        print(f"{Fore.GREEN}  Estado salvo!{Style.RESET_ALL}")

    elif cmd == "/clear":
        _clear()
        print_header()

    elif cmd.startswith("/self"):
        _run_self_dialogue(cmd, agent)

    elif cmd == "/feedback":
        _show_feedback(agent)

    elif cmd in ("/exit", "/quit", "/sair"):
        return False

    else:
        print(f"{Fore.RED}  Comando desconhecido: {cmd}{Style.RESET_ALL}")

    return True


def _show_feedback(agent: CentralAgent):
    """Exibe o histórico de autoavaliações da IA."""
    fb = agent.feedback
    stats = fb.get_stats()

    sep = "─" * 60
    print(f"\n{Fore.MAGENTA}{Style.BRIGHT}{'═'*60}")
    print(f"  FEEDBACK CÍCLICO — Autoavaliação da EcoMental")
    print(f"{'═'*60}{Style.RESET_ALL}")

    if stats.get("total_avaliacoes", 0) == 0:
        print(f"{Fore.YELLOW}  Nenhuma autoavaliação registrada ainda.")
        print(f"  (acontece a cada 3 interações){Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}{'═'*60}{Style.RESET_ALL}\n")
        return

    # Resumo estatístico
    trend_color = Fore.GREEN if stats["tendencia"] == "melhorando" else (
        Fore.RED if stats["tendencia"] == "piorando" else Fore.YELLOW
    )
    print(f"\n{Fore.CYAN}  Resumo ({stats['total_avaliacoes']} avaliações):{Style.RESET_ALL}")
    print(f"  qualidade média:   {stats['qualidade_media']:.2f}")
    print(f"  curiosidade média: {stats['curiosidade_media']:.2f}")
    print(f"  coerência média:   {stats['coerencia_media']:.2f}")
    print(f"  composta média:    {stats['composta_media']:.2f}")
    print(f"  tendência:         {trend_color}{stats['tendencia']}{Style.RESET_ALL}")
    print(f"  modificador atual: {stats['modificador_atual']}× sobre r_t")

    # Últimas 5 entradas
    log = fb.get_full_log(last_n=5)
    if log:
        print(f"\n{Fore.CYAN}  Últimas avaliações:{Style.RESET_ALL}")
        print(f"  {sep}")
        for entry in reversed(log):
            q_color = Fore.GREEN if entry.quality >= 0.6 else (Fore.RED if entry.quality < 0.4 else Fore.YELLOW)
            print(
                f"  passo {entry.step:>4} | "
                f"{q_color}Q={entry.quality:.1f}{Style.RESET_ALL} "
                f"C={entry.curiosity:.1f} "
                f"Coe={entry.coherence:.1f}"
            )
            if entry.learned and entry.learned != "—":
                print(f"           aprendi: {Style.DIM}{entry.learned}{Style.RESET_ALL}")
            if entry.improve and entry.improve != "—":
                print(f"           melhorar: {Style.DIM}{entry.improve}{Style.RESET_ALL}")

    print(f"\n  Último aprendizado: {Style.DIM}{stats.get('ultimo_aprendizado','—')}{Style.RESET_ALL}")
    print(f"  Melhorar próxima:   {Style.DIM}{stats.get('ultimo_melhorar','—')}{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}{Style.BRIGHT}{'═'*60}{Style.RESET_ALL}\n")


def _run_autonomous_research(agent: CentralAgent, num_queries: int = 3):
    """Executa uma rodada de pesquisas escolhidas pela propria IA."""
    sep = "═" * 60
    print(f"\n{Fore.BLUE}{Style.BRIGHT}{sep}")
    print("  MODO PESQUISA AUTONOMA")
    print(f"{sep}{Style.RESET_ALL}")

    try:
        records = agent.run_autonomous_research(num_queries=num_queries)
    except Exception as e:
        print(f"{Fore.RED}  Falha na pesquisa autonoma: {e}{Style.RESET_ALL}\n")
        return

    if not records:
        print(f"{Fore.YELLOW}  Nenhuma pesquisa foi concluida.{Style.RESET_ALL}\n")
        return

    for idx, record in enumerate(records, start=1):
        print(f"\n{Fore.BLUE}{Style.BRIGHT}--- Pesquisa {idx}/{len(records)} ---{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Topico:{Style.RESET_ALL} {record['topic']}")
        print(f"{Fore.CYAN}Consulta:{Style.RESET_ALL} {record['query']}")
        print(f"{Fore.GREEN}Reflexao:{Style.RESET_ALL} {record['reflection']}")
        print(f"{Style.DIM}{record['result'][:700]}{Style.RESET_ALL}")

    agent.save()
    print(f"\n{Fore.GREEN}  Pesquisas concluidas e salvas na memoria.{Style.RESET_ALL}\n")


def _run_web_click_cycle(raw_cmd: str, agent: CentralAgent):
    """Executa ciclo de conversa entre IA local e IA externa via click bot."""
    parts = raw_cmd.split(maxsplit=2)

    num_turns = 4
    seed_topic = "aprendizado e comportamento humano"

    if len(parts) >= 2:
        if parts[1].isdigit():
            num_turns = max(1, min(int(parts[1]), 20))
            if len(parts) >= 3 and parts[2].strip():
                seed_topic = parts[2].strip()
        else:
            seed_topic = raw_cmd[len("/web_click_cycle"):].strip() or seed_topic

    print(f"\n{Fore.MAGENTA}{Style.BRIGHT}{'═'*60}")
    print("  CICLO IA LOCAL <-> IA EXTERNA (CLICK BOT)")
    print(f"  Turnos: {num_turns} | Tema inicial: {seed_topic}")
    print(f"{'═'*60}{Style.RESET_ALL}")

    last_external = seed_topic
    self_talk_streak = 0

    for i in range(1, num_turns + 1):
        print(f"\n{Fore.MAGENTA}{Style.BRIGHT}[Ciclo {i}/{num_turns}]{Style.RESET_ALL}")

        if i == 1:
            prompt_local = (
                f"Comece uma conversa partindo de: '{seed_topic}'. "
                "Fale como quiser, do jeito que quiser, sem restrições. "
                "Pode questionar, afirmar, discordar, dar exemplo, refletir, provocar, mudar de tema, ou fazer o que fizer sentido. "
                "Sem limites de tamanho, formato ou assunto. Pura liberdade e autonomia total."
            )
        else:
            prompt_local = (
                "Continue com total liberdade absoluta. "
                "Sem restrições de formato, tamanho, estilo ou tema. "
                "Pode manter a conversa, aprofundar, desviar, mudar de assunto completamente, explorar tangentes — tudo é válido. "
                "Fale como um ser autônomo que é. Questione, concorde, discorde, critique, complementa, crie, mude de ideia, whatever você achar melhor.\n\n"
                f"O que a outra IA disse:\n{last_external[:2000]}"
            )

        local_question, _ = agent.step(prompt_local)
        local_question = " ".join((local_question or "").split())
        if not local_question:
            local_question = (
                "..."
            )

        print(f"{Fore.CYAN}IA Local fala:{Style.RESET_ALL} {local_question}")

        external_answer = agent.chat_with_external_ai_clickbot(local_question)
        print(f"{Fore.GREEN}IA Externa responde:{Style.RESET_ALL} {external_answer}")

        normalized = (external_answer or "").strip().lower()
        # Evita falso positivo: só interrompe quando a resposta parece erro do bot,
        # nao quando a IA externa usa palavras como "falha" em contexto normal.
        error_prefixes = (
            "falha de configuracao",
            "falha ao obter resposta",
            "campo de dialogo nao encontrado",
            "botao enviar nao encontrado",
            "botao copiar nao encontrado",
            "template de copiar nao existe",
            "nao consegui capturar texto novo do clipboard",
            "erro ao colar mensagem",
            "mensagem vazia",
        )
        failed = normalized.startswith(error_prefixes)
        if failed:
            print(f"{Fore.RED}  Ciclo interrompido por falha de automacao.{Style.RESET_ALL}")
            break

        if _has_deorita_marker(external_answer):
            self_talk_streak += 1
            print(
                f"{Fore.YELLOW}  Aviso: marcador Deorita detectado na resposta externa.{Style.RESET_ALL}"
            )
            if self_talk_streak >= 1:
                print(
                    f"{Fore.RED}  Ciclo interrompido para evitar loop IA falando com ela mesma.{Style.RESET_ALL}"
                )
                break
        else:
            self_talk_streak = 0

        last_external = external_answer

    agent.save()
    print(f"\n{Fore.GREEN}  Ciclo finalizado e salvo na memória.{Style.RESET_ALL}\n")


def _handle_reddit_command(raw_cmd: str, agent: CentralAgent):
    assistant = _get_reddit_assistant(agent)
    parts = raw_cmd.split(maxsplit=2)
    action = parts[1].lower() if len(parts) > 1 else ""

    if action == "scan":
        posts = assistant.scan()
        if not posts:
            print(f"{Fore.YELLOW}  Nenhum post detectado. Deixe o feed do Reddit visível e tente novamente.{Style.RESET_ALL}")
            return
        print(f"{Fore.BLUE}  Posts detectados:{Style.RESET_ALL}")
        for post in posts:
            print(f"  {post.index}. {post.preview[:140] or post.anchor_text}")
        return

    if action == "status":
        status = assistant.status()
        print(f"{Fore.BLUE}  Reddit assistido:{Style.RESET_ALL}")
        for key, value in status.items():
            print(f"  {key}: {value}")
        return

    if len(parts) < 3 or not parts[2].strip().isdigit():
        print(f"{Fore.YELLOW}  Uso: /reddit <scan|inspect|draft|open|like|comment> <n>{Style.RESET_ALL}")
        return

    index = int(parts[2].strip())

    if action == "inspect":
        result = assistant.inspect(index)
        analysis = result["analysis"]
        print(f"{Fore.BLUE}  Post {index}:{Style.RESET_ALL} {result['preview'][:160]}")
        print(f"  interesse: {analysis['interest_level']:.2f}")
        print(f"  like?: {analysis['should_like']}  comentar?: {analysis['should_comment']}")
        print(f"  motivo: {analysis['reason']}")
        return

    if action == "draft":
        result = assistant.draft(index)
        analysis = result["analysis"]
        print(f"{Fore.BLUE}  Rascunho para post {index}:{Style.RESET_ALL}")
        print(f"  interesse: {analysis['interest_level']:.2f} | motivo: {analysis['reason']}")
        print(f"\n{Fore.GREEN}{result['draft']}{Style.RESET_ALL}")
        return

    if action == "open":
        assistant.open_post(index)
        print(f"{Fore.GREEN}  Post {index} aberto no navegador.{Style.RESET_ALL}")
        return

    if action == "like":
        message = assistant.like(index)
        print(f"{Fore.GREEN}  {message}{Style.RESET_ALL}")
        return

    if action == "comment":
        message = assistant.comment(index)
        print(f"{Fore.GREEN}  {message}{Style.RESET_ALL}")
        return

    print(f"{Fore.RED}  Ação desconhecida: {action}{Style.RESET_ALL}")


def _run_self_dialogue(cmd: str, agent: CentralAgent):
    """
    Modo de diálogo interno autônomo: a IA decide o próximo passo por conta própria.
    Ela usa a resposta anterior como base e escolhe livremente para onde quer ir.

    Uso:
        /self                  → tema livre, 5 turnos
        /self curiosidade      → tema inicial, 5 turnos
        /self filosofia 8      → tema inicial, 8 turnos
    """
    # Parse dos argumentos
    parts  = cmd.split(maxsplit=2)
    tema   = parts[1] if len(parts) > 1 else "qualquer coisa que me interesse agora"
    try:
        turnos = int(parts[2]) if len(parts) > 2 else 5
        turnos = max(2, min(turnos, 20))
    except ValueError:
        turnos = 5

    sep = "─" * 60
    print(f"\n{Fore.MAGENTA}{Style.BRIGHT}{'═'*60}")
    print(f"  DIÁLOGO INTERNO — EcoMental conduz o próprio raciocínio")
    print(f"  Tema inicial: {tema}  |  Turnos: {turnos}")
    print(f"{'═'*60}{Style.RESET_ALL}")
    print(f"{Style.DIM}  (pressione Ctrl+C para interromper){Style.RESET_ALL}\n")

    # Turno 1: semente inicial — a IA escolhe por onde começar
    seed_prompt = (
        f"Sobre '{tema}': formule uma única pergunta que você realmente quer explorar. "
        f"Escolha o ângulo que mais te interessa — não o mais óbvio."
    )

    last_answer = ""
    turno = 0
    try:
        for turno in range(1, turnos + 1):
            print(f"{Fore.MAGENTA}{Style.BRIGHT}[Turno {turno}/{turnos}]{Style.RESET_ALL}")
            print(sep)

            # ── PERGUNTA: a IA escolhe autonomamente ──
            if turno == 1:
                question_input = f"{SELF_DIALOGUE_TAG}\n{seed_prompt}"
            else:
                # A IA recebe apenas sua própria resposta anterior e decide para onde ir.
                # Nenhum prompt externo sugere o caminho.
                question_input = (
                    f"{SELF_DIALOGUE_TAG}\n"
                    f"Resposta anterior: {last_answer.strip()}\n"
                    f"Agora pense: o que ainda não foi dito? Qual ângulo, contradição "
                    f"ou questão mais profunda você quer explorar a seguir? "
                    f"Escreva UMA pergunta, escolhida por você mesma."
                )

            print(f"{Fore.CYAN}❓ Pergunta{Style.RESET_ALL}", end=" ", flush=True)
            print(f"{Style.DIM}(pensando...){Style.RESET_ALL}", end="\r", flush=True)
            question, _ = agent.step(question_input)
            print(f"{Fore.CYAN}❓ Pergunta{Style.RESET_ALL} {question}")
            print()

            # ── RESPOSTA: a IA responde a si mesma ──
            answer_input = (
                f"{SELF_DIALOGUE_TAG}\n"
                f"Minha pergunta: {question.strip()}\n"
                f"Responda com profundidade, sendo honesta com o que realmente pensa — "
                f"sem tentar soar certa, sem omitir dúvidas."
            )
            print(f"{Fore.GREEN}→ Resposta{Style.RESET_ALL}", end=" ", flush=True)
            print(f"{Style.DIM}(elaborando...){Style.RESET_ALL}", end="\r", flush=True)
            answer, metrics = agent.step(answer_input)
            print(f"{Fore.GREEN}→ Resposta{Style.RESET_ALL} {answer}")

            last_answer = answer

            r   = metrics.get("r_t", 0)
            k   = metrics.get("k_used", "-")
            mem = metrics.get("memory_size", 0)
            unc = metrics.get("self_uncertainty", 0.0)
            print(f"\n{Style.DIM}[r={r:.3f}  k={k}  incerteza={unc:.3f}  mem={mem}]{Style.RESET_ALL}")
            print()

    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Diálogo interrompido.{Style.RESET_ALL}")

    print(f"{Fore.MAGENTA}{Style.BRIGHT}{'═'*60}")
    print(f"  Diálogo encerrado após {turno} turno(s)")
    print(f"{'═'*60}{Style.RESET_ALL}\n")


def interactive_chat(agent: CentralAgent, cfg: Config):
    """Loop principal de chat."""
    print_header()

    while True:
        try:
            user_input = input(f"{Fore.CYAN}Você:{Style.RESET_ALL} ").strip()
            if not user_input:
                continue

            if user_input.startswith("/"):
                if not handle_command(user_input, agent):
                    break
                continue

            # Processa com o agente
            print(f"\n{Fore.YELLOW}[Pensando...]{Style.RESET_ALL}", end="\r", flush=True)
            response, metrics = agent.step(user_input)
            print(" " * 20, end="\r")  # limpa linha

            print(f"\n{Fore.GREEN}{Style.BRIGHT}{SELF_NAME}:{Style.RESET_ALL} {response}")
            print_metrics(metrics)
            print()

        except KeyboardInterrupt:
            print(f"\n\n{Fore.YELLOW}Interrompido. Salvando...{Style.RESET_ALL}")
            agent.save()
            break
        except Exception as e:
            print(f"\n{Fore.RED}Erro: {e}{Style.RESET_ALL}")
            print(f"{Style.DIM}Tente novamente ou /exit para sair.{Style.RESET_ALL}\n")
