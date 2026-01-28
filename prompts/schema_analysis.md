{
    "type": "OBJECT",
    "properties": {
        "logline": {
            "type": "STRING",
            "description": "Основная идея произведения в одном-двух предложениях."
        },
        "key_characters": {
            "type": "ARRAY",
            "description": "Список ключевых персонажей и их краткие роли в сюжете.",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "name": {
                        "type": "STRING",
                        "description": "Имя персонажа."
                    },
                    "role": {
                        "type": "STRING",
                        "description": "Краткое описание роли персонажа (протагонист, антагонист, ментор и т.д.)."
                    }
                },
                "required": ["name", "role"]
            }
        },
        "setting": {
            "type": "STRING",
            "description": "Описание сеттинга: время и место действия."
        },
        "main_conflict": {
            "type": "STRING",
            "description": "Описание основного конфликта произведения."
        },
        "plot_structure": {
            "type": "STRING",
            "description": "Предполагаемая структура сюжета (например, трёхактная структура, путешествие героя)."
        },
        "themes_and_motifs": {
            "type": "ARRAY",
            "description": "Список ключевых тем и мотивов.",
            "items": {
                "type": "STRING"
            }
        },
        "recurring_secondary_characters": {
            "type": "ARRAY",
            "description": "Список второстепенных персонажей, которые появляются неоднократно.",
            "items": {
                "type": "STRING"
            }
        },
        "originality_check": {
            "type": "OBJECT",
            "description": "Проверка на оригинальность и избегание клише.",
            "properties": {
                "avoids_obvious_solutions": {
                    "type": "BOOLEAN",
                    "description": "Избегает ли сюжет очевидных решений (месть, тайный агент и т.д.)."
                },
                "dual_role_character_present": {
                    "type": "BOOLEAN",
                    "description": "Присутствует ли персонаж с двойной ролью."
                },
                "plot_twists": {
                    "type": "STRING",
                    "description": "Описание 1-2 неожиданных, но достоверных сюжетных поворотов."
                }
            },
            "required": ["avoids_obvious_solutions", "dual_role_character_present", "plot_twists"]
        },
        "detailed_plot_plan": {
            "type": "STRING",
            "description": "Развёрнутый поглавный план сюжета с чёткой хронологией."
        },
        "genre_expectations_check": {
            "type": "STRING",
            "description": "Анализ соответствия жанровым ожиданиям и наличия неожиданных элементов."
        },
        "parallel_storylines": {
            "type": "OBJECT",
            "description": "Описание параллельных сюжетных линий.",
            "properties": {
                "subplots": {
                    "type": "ARRAY",
                    "description": "Список подсюжетов.",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "type": {
                                "type": "STRING",
                                "description": "Тип подсюжета (Личная арка героя, Развитие отношений, Тайна прошлого)."
                            },
                            "description": {
                                "type": "STRING",
                                "description": "Краткое описание подсюжета."
                            }
                        },
                        "required": ["type", "description"]
                    }
                },
                "interweaving_rules": {
                    "type": "STRING",
                    "description": "Анализ того, как подсюжеты переплетаются с основным сюжетом."
                }
            },
            "required": ["subplots", "interweaving_rules"]
        }
    },
    "required": [
        "logline",
        "key_characters",
        "setting",
        "main_conflict",
        "plot_structure",
        "themes_and_motifs",
        "recurring_secondary_characters",
        "originality_check",
        "detailed_plot_plan",
        "genre_expectations_check",
        "parallel_storylines"
    ]
}
