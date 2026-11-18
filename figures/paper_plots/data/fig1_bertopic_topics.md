# Game Review BERTopic

- Generated: 2026-07-01T05:12:17
- Input H5: `game_review_data/embedding_h5.h5`
- Corpus format: `embedding_h5`
- H5 games scanned: 2000
- H5 size: 152.912 GB
- Sample method: `balanced_prefix_per_game`
- Sampled documents: 100000
- Embedding dimension: 1024
- UMAP `n_neighbors`: 15
- HDBSCAN `min_cluster_size`: 50
- CountVectorizer `min_df`: 1
- Random state: 42
- Fit time: 85.9 seconds

Note: this run uses a deterministic per-game prefix sample and skips metadata records with `review_id < 3` before fitting BERTopic. It is a practical topic snapshot, not a full all-sentence HDBSCAN fit.

## Environment

| Package | Version |
|---|---:|
| `bertopic` | `0.17.4` |
| `hdbscan` | `0.8.44` |
| `umap-learn` | `0.5.12` |
| `h5py` | `3.15.1` |
| `numpy` | `2.4.3` |
| `scikit-learn` | `1.8.0` |

## Summary

- Topics excluding outliers: 127
- Outlier documents: 54898
- Outlier rate: 54.90%

## Topic Table

| Topic | Count | Top Words |
|---:|---:|---|
| -1 | 54898 | outliers |
| 0 | 15006 | bug, boss, npc, bossboss, boss boss, steam, npc npc, debuff, bug bug, spoiler |
| 1 | 1474 | hours, horas, hour, minutes, played, took, hours game, game hours, played hours, took hours |
| 2 | 1362 | bad, say, im, thats, know, wrong, thing, dont, just, things |
| 3 | 1244 | multiplayer, coop, online, friends, servers, play, mode, server, players, singleplayer |
| 4 | 1104 | steam, ea, steam deck, deck, launcher, epic, game steam, launch, origin, account |
| 5 | 875 | game, love game, love, playing, loved, enjoyed, like game, played, really, wanted |
| 6 | 872 | music, soundtrack, sound, sound design, sounds, sonora, audio, tracks, design, trilha |
| 7 | 702 | price, worth, sale, buy, promoo, worth price, preo, buying, espere, espere uma |
| 8 | 631 | devcats, iron lung, lung, lung iron, iron, xy, qte, cats, lowend, gestalt |
| 9 | 624 | le, et, jeu, les, pas, je, des, pour, qui, ce |
| 10 | 569 | fps, rtx, settings, gtx, dlss, gpu, ram, nvidia, cpu, pc |
| 11 | 560 | combat, combate, battles, fun, fun combat, combat fun, combat feels, game combat, feels, battle |
| 12 | 516 | dlc, dlcs, dlc dlc, base game, content, paid dlc, base, edition, paid, game dlc |
| 13 | 508 | graphics, art, pixel, grficos, pixel art, style, grafik, art style, beautiful, los |
| 14 | 473 | price, sale, worth, preo, jogo, price tag, tag, game sale, reais, game |
| 15 | 455 | la, pizza, chocolate, el, minutos, overcooked, taza, bowl, steak, step |
| 16 | 453 | car, cars, driving, drive, truck, vehicles, grip, wheel, brake, carro |
| 17 | 433 | map, maps, mapa, mapas, world map, explore, location, areas, world, map map |
| 18 | 419 | devs, developers, dev, developer, team, community, feedback, dev team, work, listen |
| 19 | 418 | controller, controls, mouse, keyboard, button, controle, xbox, buttons, press, wasd |
| 20 | 412 | review, reviews, negative, negative review, write, write review, update review, read, original review, negative reviews |
| 21 | 402 | weapons, weapon, ammo, guns, gun, rifle, shotgun, rifles, melee, shotguns |
| 22 | 395 | review, recommend, recommend game, reviews, negative, negative reviews, game, review game, positive, negative review |
| 23 | 389 | quake, house flipper, flipper, quake quake, house, champions, overcooked, nintendo, h1, spoiler |
| 24 | 385 | roguelike, rogue, doom, roguelite, rpg, souls, diablo, skyrim, dungeon, like |
| 25 | 378 | simulator, stardew, sims, stardew valley, sim, valley, walking, farming, walking simulator, simulation |
| 26 | 376 | horror, horror game, resident evil, resident, evil, survival horror, scary, survival, scares, horror games |
| 27 | 369 | translation, english, language, chinese, traduo, translate, bersetzung, bersetzt, idioma, localization |
| 28 | 361 | puzzles, puzzle, puzzle game, solving, solve, puzzle games, easy, picross, puzzles game, riddles |
| 29 | 348 | story, ending, plot, interesting, story story, end, storytelling, characters, emotional, narrative |
| 30 | 310 | enemies, enemy, attack, dodge, attacks, hit, cover, block, enemy types, enemy attacks |
| 31 | 303 | animals, cat, dogs, animal, pet, dog, cats, horse, pets, animais |
| 32 | 292 | mod, chinese, bug, milk1, tiny bunny, deeply, swipe, chinese version, adv, translated |
| 33 | 292 | early access, access, early, game early, access game, release, alpha, released, access early, access release |
| 34 | 265 | houses, build, building, buildings, village, furniture, house, bricks, placed, base |
| 35 | 262 | enjoyed, fun, enjoying, really, like, really enjoyed, enjoy, liked, enjoyed time, did |
| 36 | 260 | mods, mod, modding, daggerfall, modding community, dot, daggerfall unity, community, unity, modded |
| 37 | 251 | characters, character, main, memorable, personajes, protagonists, endearing, suffering builds, characters characters, builds character |
| 38 | 247 | ai, pathing, bots, ia, ai ai, pathfinding, ai just, ai doesnt, la ia, ai really |
| 39 | 236 | difficulty, hard, difficulties, difficult, difficulty settings, harder, levels, challenging, hard mode, challenge |
| 40 | 235 | good game, great game, good, amazing, game, great, game amazing, game great, game game, amazing game |
| 41 | 218 | really really, recommend, recommend game, highly, really, highly recommend, game recommend, game highly, game, definitely recommend |
| 42 | 210 | bugs, bug, updates, patch, patches, update, fixed, fix, minor, issues |
| 43 | 209 | quests, quest, sidequests, main, complete, main story, story quest, fetch, story, main quest |
| 44 | 207 | ship, ships, fishing, boat, fleet, boats, sea, minigame, underwater, atlantic |
| 45 | 206 | recommend, recommended, highly, recommendation, highly recommended, recommend recommend, highly recommend, dont recommend, je, recommend highly |
| 46 | 202 | pc, port, version, pc port, pc version, ps4, console, consoles, gen, ps2 |
| 47 | 198 | ai2lucy, ai2lucy got, city final, got problems, edition ai2lucy, final edition, scarlet maidenmaid, milfy city, maidenmaid mansion, problems scarlet |
| 48 | 198 | discord, banned, ban, community, bans, forums, discord server, permanently, server, post |
| 49 | 195 | rating, overall, rate, solid, overall game, nota, rate game, recommend, id, nota para |
| 50 | 193 | remaster, remake, remastered, original, remasters, el, que, original game, mafia, version |
| 51 | 193 | issues, game, mechanics, unfortunately, biggest, problem, issues game, problems game, gameplay, lot |
| 52 | 192 | vr, vr games, vr game, best vr, vrchat, headset, vr vr, cvr, best, games |
| 53 | 188 | cards, card, deck, decks, deckbuilding, draw, cards deck, trading cards, card games, hand |
| 54 | 188 | crafting, craft, inventory, materials, item, items, resources, storage, gear, craft item |
| 55 | 187 | missions, mission, complete, misiones, missions missions, rewards, misses, campaign, missions feel, main missions |
| 56 | 183 | favorite, best, favorite games, played, favorite game, favourite, games, best games, ive played, best game |
| 57 | 172 | voice, voice acting, acting, actors, voice actors, voices, voiced, acted, characters, voice acted |
| 58 | 168 | issues, problems, problem, issue, needs, things, algumas, lot work, maybe lot, foram |
| 59 | 164 | skills, skill, skill points, points, level, character, tree, levels, gear, passive |
| 60 | 159 | bosses, boss, boss fights, fights, final boss, fight, boss fight, final, challenging, bosses game |
| 61 | 158 | money, dinheiro, waste, waste money, sobrando, se tiver, dinheiro sobrando, compre se, debt, tiver dinheiro |
| 62 | 149 | save, saves, save game, save file, autosave, file, saved, progress, manual, manual save |
| 63 | 146 | plant, crops, grow, crop, farming, plants, ready, harvest, seeds, lets |
| 64 | 142 | speed limit, limit, speed, locomotive, railway, lcd, car, ui, signal, arrow |
| 65 | 135 | good decent, bad, decent bad, beautiful good, good, decent, beautiful, bad beautiful, bad bad, schlecht |
| 66 | 135 | needs, hope, future, suggestions, potential, game, forward, things need, game potential, game better |
| 67 | 134 | abandoned, updates, development, devs, update, developers, developer, game, players, dev |
| 68 | 133 | dishes, cooking, cleaning, customers, food, clean, toilet, restaurant, ingredients, flip |
| 69 | 132 | humor, jokes, funny, laugh, humour, joke, make laugh, hilarious, silly, references |
| 70 | 123 | bugs, bugs game, game breaking, bug, gamebreaking, buggy, breaking, issues, breaking bugs, mnh |
| 71 | 121 | achievements, achievement, score, leaderboardsranks isnt, high score, care leaderboardsranks, leaderboardsranks, isnt necessary, necessary progress, achievements wait |
| 72 | 121 | animations, animation, animaes, animated, smooth, theyre, animations bit, animations look, sprites, stiff |
| 73 | 121 | human fall, oa, fall flat, oo, flat, fall, human |
| 74 | 120 | art, art style, style, arte, artistic, visuals, artstyle, beautiful, painting, work art |
| 75 | 119 | check run, run paint, paint, requirements check, check, pc requirements, requirements, teens, kids teens, teens adults |
| 76 | 119 | wa, open eyes, ga, eyes, logic open, wo suru, suru, wo, pole, pal |
| 77 | 118 | ui, interface, menus, ui ui, menu, ui elements, sampai, user interface, dalam, user |
| 78 | 118 | xcom, xcoms, xcom2, reboot xcoms, reboot, like xcom, phoenix point, strategy, phoenix, tactics |
| 79 | 112 | nasa, nasa spare, ask nasa, spare computer, spare, computer, ask, computer difficulty, just press, press |
| 80 | 112 | dark souls, dark, souls, grind, souls grind, difficult dark, grind grind, difficult, grind difficult, grind dark |
| 81 | 106 | pvp, pve, pvp pvp, pve pvp, pvp f2p, server, gzw, consegu, pvpve, pvp game |
| 82 | 96 | fnaf, birthday, stayed, witch, fish, happy, came, nice, love, gameplay |
| 83 | 94 | fell, death, die, died, dead, lyorum, ento, burried, exato momento, exato |
| 84 | 90 | lumina, nazi, beloved, award, start new, porn, wife, buy game, thank, thought |
| 85 | 90 | chinesewe, chinesewe need, need chinesewe, need, need chinese, chinese, chinese need, besoin, chinois, avons besoin |
| 86 | 90 | demons, werewolf, named, night shifts, protagonist, kyle, amanda, curse, ritual, job |
| 87 | 86 | hard master, learn hard, easy learn, master, master easy, learn, easy, hard, aprender, fcil |
| 88 | 85 | free, free play, game free, freetoplay, free game, play, game freetoplay, base game, game, f2p |
| 89 | 82 | mullet madjack, mason lindroth, mullet, madjack, flatout, mason, carnage, limbo, upd, god eater |
| 90 | 80 | simple, easy, game simple, gameplay, gameplay simple, simple game, loop, gameplay loop, mechanics, complex |
| 91 | 80 | bugs, experienced, issues, optimization, crashes, optimized, glitches, havent experienced, encountered, crash |
| 92 | 79 | metroidvania, metroid, metroidvanias, megaman, metal gear, genre, titulo, hollow knight, metal, msx |
| 93 | 78 | release franchise, franchise release, franchise, release, shrine gain, offer shrine, shrine, gain offer, gain, offer |
| 94 | 77 | keyboard, player keyboard, escadd, escadd player, xbox, player, pc, emmm, keyboard player, assign keyboard |
| 95 | 77 | oo, icebreakerlust, touchthe, powerliving kanazawarosewater, kanazawarosewater manor88name88s, icebreakerlust powerliving, manor88name88s triangle, manor88name88s, kanazawarosewater, milky touchthe |
| 96 | 77 | infinity, infinity price, long infinity, average long, price, average, calculadora, com pilha, uma calculadora, pilha |
| 97 | 75 | dry instead, watch paint, paint dry, dry, paint, watch, midnight paradise, instead watch, midnight, paradise |
| 98 | 74 | trailer, trailers, sequel, series, fan series, srie, franchise, game look, series id, waiting years |
| 99 | 74 | campaign, campanha, campaigns, imperialism, kampagne, just campaign, campaign does, endgame content, verschiedene, scripted |
| 100 | 73 | developers, dev, developer, devs, developed, person, game, game developed, studios, small |
| 101 | 72 | level design, design, level, levels, designed, design good, game level, games level, design bit, great level |
| 102 | 72 | crashes, crash, crashing, game crashes, game crash, crashed, crash game, random, screen, constantly crashes |
| 103 | 71 | camera, angle, view, camera angle, camra, screws, cameras, fixed camera, angles, camra est |
| 104 | 68 | platforming, platformer, platformers, sections, platforming sections, puzzles, challenging, platformer sections, good platformer, fights |
| 105 | 67 | madrid, und ich, spieler, und, mick, mannschaft, meine mannschaft, der, nicht weg, caej |
| 106 | 65 | environment, environments, atmosphere, beautiful, scenery, etkileim, atmosphere just, ne ok, vistas, stunning |
| 107 | 65 | p5, djmax, psp, ps5, reborn, ps3, enhanced, persona, ps, version |
| 108 | 65 | winning son, winning, psychopath, psychopath psychopath, son winning, son, ya winning, ya, sir, stoic |
| 109 | 65 | total war, total, war, rome, rts, war games, shogun, warhammer, strategy, war game |
| 110 | 63 | tutorial, tutorials, das tutorial, das, habe ich, los tutoriales, content doing, tutorials tutorial, tutorial single, tutoriales |
| 111 | 62 | graphics forget, forget reality, reality, reality graphics, forget, graphics, voc esquece, esquece, esquece que, que realidade |
| 112 | 61 | little nightmares, northern journey, northern, nightmares, journey, npc, little, games |
| 113 | 59 | replayability, replay, replay value, value, replayable, challenges, good replayability, difficulty, different, add |
| 114 | 58 | janken, isekai janken, simulator fusion, isekai, janken hero, fusion, hero simulator, fusion isekai, hero, simulator |
| 115 | 58 | cruub, cruub cruub, king cruub, cruub king, monke, king, wren, god, jr, ya |
| 116 | 58 | gmod, gmod gmod, gb, gb gb, exe gmod, gmod exe, exe, g50100, g50100 g100200, gb gmod |
| 117 | 58 | jump, jumps, jumping, wall, fall, crystal arrow, momentum, double, slide, walls |
| 118 | 57 | bug bug, bug, dedsec, bgm bug, bug bugbug, game joke, fk, game strongly, preferably, fh |
| 119 | 56 | chinese need, need chinese, chinese, pleasekorean, pleasekorean pleasekorean, need, china, republic, scenario, war ii |
| 120 | 56 | visual, visual novel, novel, novels, visual novels, digimon, musicals, novel game, story, great visual |
| 121 | 55 | character, notre, create character, customize, mask, customize character, body, gender, customization, changer |
| 122 | 54 | significant brain, brain usage, brain, usage, significant, usage significant, uso significativo, significativo, uso, significativo crebro |
| 123 | 53 | fun game, fun, really fun, game fun, lot fun, game lot, damn fun, game damn, genuinely fun, extremely fun |
| 124 | 53 | buy game, worth, game worth, buy, pena, vale, thinking buying, comprar, definitely worth, game definitely |
| 125 | 52 | eargasm good, bad bad, eargasm, good bad, bad, audio eargasm, dont audio, good good, audio, just dont |
| 126 | 52 | rich ladys, ladys slave, slave role, leaderboardsranks, care leaderboardsranks, ladys, role play, slave, rich, role |
