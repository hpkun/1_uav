# Plain transition mechanism diagnostic closure audit

## Protocol
Plain MAPPO seeds 5301/5302/5303 at 3M final; deterministic 44M cases only.

## Replay integrity
`DIAGNOSTIC_REPLAY_MATCHED`

## Counts
Direct=150, native transitions=232, canonical transitions=232, native continuations=696, canonical continuations=696.

## Death precedence impact
Count=0; `ZERO_REALIZED_IMPACT`.

## Case outcomes
{
  "all_same_waves": 9,
  "all_zero": 0,
  "all_three": 5,
  "5303_strictly_better_than_both": 11,
  "5303_unique_three_wave": 9,
  "5301_or_5302_three_but_5303_not": 8,
  "disagreement": {
    "5301_vs_5302": 33,
    "5301_vs_5303": 29,
    "5302_vs_5303": 29
  },
  "5303_minus_others_distribution": {},
  "5303_minus_5301": {
    "-3": 1,
    "-2": 1,
    "-1": 6,
    "0": 21,
    "1": 7,
    "2": 9,
    "3": 5
  },
  "5303_minus_5302": {
    "-3": 0,
    "-2": 1,
    "-1": 5,
    "0": 21,
    "1": 8,
    "2": 13,
    "3": 2
  },
  "mean_paired_waves_difference": {
    "5303-5301": 0.58,
    "5303-5302": 0.66
  },
  "most_discriminating_cases": [
    {
      "evaluation_seed": 44000048,
      "waves_5301": 3,
      "waves_5302": 0,
      "waves_5303": 3
    },
    {
      "evaluation_seed": 44000044,
      "waves_5301": 3,
      "waves_5302": 0,
      "waves_5303": 2
    },
    {
      "evaluation_seed": 44000043,
      "waves_5301": 0,
      "waves_5302": 1,
      "waves_5303": 3
    },
    {
      "evaluation_seed": 44000041,
      "waves_5301": 0,
      "waves_5302": 3,
      "waves_5303": 3
    },
    {
      "evaluation_seed": 44000036,
      "waves_5301": 0,
      "waves_5302": 3,
      "waves_5303": 3
    },
    {
      "evaluation_seed": 44000030,
      "waves_5301": 0,
      "waves_5302": 1,
      "waves_5303": 3
    },
    {
      "evaluation_seed": 44000013,
      "waves_5301": 0,
      "waves_5302": 2,
      "waves_5303": 3
    },
    {
      "evaluation_seed": 44000011,
      "waves_5301": 3,
      "waves_5302": 0,
      "waves_5303": 3
    },
    {
      "evaluation_seed": 44000006,
      "waves_5301": 3,
      "waves_5302": 0,
      "waves_5303": 2
    },
    {
      "evaluation_seed": 44000005,
      "waves_5301": 3,
      "waves_5302": 1,
      "waves_5303": 0
    }
  ],
  "5303_three_while_both_others_at_most_one": 6
}

## Clear-conditioned survivors
[
  {
    "policy_seed": 5301,
    "wave": 1,
    "record_N": 50,
    "conditional_on_record": 2.84,
    "clear_N": 43,
    "conditional_on_clear": 3.302325581395349
  },
  {
    "policy_seed": 5301,
    "wave": 2,
    "record_N": 43,
    "conditional_on_record": 2.1627906976744184,
    "clear_N": 27,
    "conditional_on_clear": 3.074074074074074
  },
  {
    "policy_seed": 5301,
    "wave": 3,
    "record_N": 27,
    "conditional_on_record": 1.8148148148148149,
    "clear_N": 20,
    "conditional_on_clear": 2.45
  },
  {
    "policy_seed": 5302,
    "wave": 1,
    "record_N": 50,
    "conditional_on_record": 2.8,
    "clear_N": 43,
    "conditional_on_clear": 3.255813953488372
  },
  {
    "policy_seed": 5302,
    "wave": 2,
    "record_N": 43,
    "conditional_on_record": 1.9534883720930232,
    "clear_N": 29,
    "conditional_on_clear": 2.896551724137931
  },
  {
    "policy_seed": 5302,
    "wave": 3,
    "record_N": 29,
    "conditional_on_record": 1.206896551724138,
    "clear_N": 14,
    "conditional_on_clear": 2.5
  },
  {
    "policy_seed": 5303,
    "wave": 1,
    "record_N": 50,
    "conditional_on_record": 3.24,
    "clear_N": 48,
    "conditional_on_clear": 3.375
  },
  {
    "policy_seed": 5303,
    "wave": 2,
    "record_N": 48,
    "conditional_on_record": 2.5625,
    "clear_N": 42,
    "conditional_on_clear": 2.9285714285714284
  },
  {
    "policy_seed": 5303,
    "wave": 3,
    "record_N": 42,
    "conditional_on_record": 1.8095238095238095,
    "clear_N": 29,
    "conditional_on_clear": 2.6206896551724137
  }
]

## Native entry effects
{
  "wave2": [
    {
      "next_wave": 2,
      "continuation_policy": 5301,
      "contrast": "source5303-source5301",
      "N": 36,
      "wins": 8,
      "losses": 7,
      "ties": 21,
      "mean_paired_outcome_delta": 0.027777777777777776
    },
    {
      "next_wave": 2,
      "continuation_policy": 5301,
      "contrast": "source5303-source5302",
      "N": 36,
      "wins": 7,
      "losses": 6,
      "ties": 23,
      "mean_paired_outcome_delta": 0.027777777777777776
    },
    {
      "next_wave": 2,
      "continuation_policy": 5302,
      "contrast": "source5303-source5301",
      "N": 36,
      "wins": 8,
      "losses": 7,
      "ties": 21,
      "mean_paired_outcome_delta": 0.027777777777777776
    },
    {
      "next_wave": 2,
      "continuation_policy": 5302,
      "contrast": "source5303-source5302",
      "N": 36,
      "wins": 9,
      "losses": 7,
      "ties": 20,
      "mean_paired_outcome_delta": 0.05555555555555555
    },
    {
      "next_wave": 2,
      "continuation_policy": 5303,
      "contrast": "source5303-source5301",
      "N": 36,
      "wins": 4,
      "losses": 5,
      "ties": 27,
      "mean_paired_outcome_delta": -0.027777777777777776
    },
    {
      "next_wave": 2,
      "continuation_policy": 5303,
      "contrast": "source5303-source5302",
      "N": 36,
      "wins": 7,
      "losses": 5,
      "ties": 24,
      "mean_paired_outcome_delta": 0.05555555555555555
    }
  ],
  "wave3": [
    {
      "next_wave": 3,
      "continuation_policy": 5301,
      "contrast": "source5303-source5301",
      "N": 16,
      "wins": 6,
      "losses": 0,
      "ties": 10,
      "mean_paired_outcome_delta": 0.375
    },
    {
      "next_wave": 3,
      "continuation_policy": 5301,
      "contrast": "source5303-source5302",
      "N": 16,
      "wins": 4,
      "losses": 0,
      "ties": 12,
      "mean_paired_outcome_delta": 0.25
    },
    {
      "next_wave": 3,
      "continuation_policy": 5302,
      "contrast": "source5303-source5301",
      "N": 16,
      "wins": 2,
      "losses": 2,
      "ties": 12,
      "mean_paired_outcome_delta": 0.0
    },
    {
      "next_wave": 3,
      "continuation_policy": 5302,
      "contrast": "source5303-source5302",
      "N": 16,
      "wins": 5,
      "losses": 4,
      "ties": 7,
      "mean_paired_outcome_delta": 0.0625
    },
    {
      "next_wave": 3,
      "continuation_policy": 5303,
      "contrast": "source5303-source5301",
      "N": 16,
      "wins": 3,
      "losses": 6,
      "ties": 7,
      "mean_paired_outcome_delta": -0.1875
    },
    {
      "next_wave": 3,
      "continuation_policy": 5303,
      "contrast": "source5303-source5302",
      "N": 16,
      "wins": 7,
      "losses": 3,
      "ties": 6,
      "mean_paired_outcome_delta": 0.25
    }
  ]
}

## Canonical entry effects
{
  "wave2": [
    {
      "next_wave": 2,
      "continuation_policy": 5301,
      "contrast": "source5303-source5301",
      "N": 36,
      "wins": 8,
      "losses": 6,
      "ties": 22,
      "mean_paired_outcome_delta": 0.05555555555555555
    },
    {
      "next_wave": 2,
      "continuation_policy": 5301,
      "contrast": "source5303-source5302",
      "N": 36,
      "wins": 7,
      "losses": 5,
      "ties": 24,
      "mean_paired_outcome_delta": 0.05555555555555555
    },
    {
      "next_wave": 2,
      "continuation_policy": 5302,
      "contrast": "source5303-source5301",
      "N": 36,
      "wins": 8,
      "losses": 5,
      "ties": 23,
      "mean_paired_outcome_delta": 0.08333333333333333
    },
    {
      "next_wave": 2,
      "continuation_policy": 5302,
      "contrast": "source5303-source5302",
      "N": 36,
      "wins": 9,
      "losses": 4,
      "ties": 23,
      "mean_paired_outcome_delta": 0.1388888888888889
    },
    {
      "next_wave": 2,
      "continuation_policy": 5303,
      "contrast": "source5303-source5301",
      "N": 36,
      "wins": 6,
      "losses": 4,
      "ties": 26,
      "mean_paired_outcome_delta": 0.05555555555555555
    },
    {
      "next_wave": 2,
      "continuation_policy": 5303,
      "contrast": "source5303-source5302",
      "N": 36,
      "wins": 7,
      "losses": 4,
      "ties": 25,
      "mean_paired_outcome_delta": 0.08333333333333333
    }
  ],
  "wave3": [
    {
      "next_wave": 3,
      "continuation_policy": 5301,
      "contrast": "source5303-source5301",
      "N": 16,
      "wins": 4,
      "losses": 1,
      "ties": 11,
      "mean_paired_outcome_delta": 0.1875
    },
    {
      "next_wave": 3,
      "continuation_policy": 5301,
      "contrast": "source5303-source5302",
      "N": 16,
      "wins": 6,
      "losses": 1,
      "ties": 9,
      "mean_paired_outcome_delta": 0.3125
    },
    {
      "next_wave": 3,
      "continuation_policy": 5302,
      "contrast": "source5303-source5301",
      "N": 16,
      "wins": 4,
      "losses": 1,
      "ties": 11,
      "mean_paired_outcome_delta": 0.1875
    },
    {
      "next_wave": 3,
      "continuation_policy": 5302,
      "contrast": "source5303-source5302",
      "N": 16,
      "wins": 6,
      "losses": 5,
      "ties": 5,
      "mean_paired_outcome_delta": 0.0625
    },
    {
      "next_wave": 3,
      "continuation_policy": 5303,
      "contrast": "source5303-source5301",
      "N": 16,
      "wins": 4,
      "losses": 4,
      "ties": 8,
      "mean_paired_outcome_delta": 0.0
    },
    {
      "next_wave": 3,
      "continuation_policy": 5303,
      "contrast": "source5303-source5302",
      "N": 16,
      "wins": 3,
      "losses": 4,
      "ties": 9,
      "mean_paired_outcome_delta": -0.0625
    }
  ]
}

## Controller effects
{
  "native": {
    "wave2": [
      {
        "next_wave": 2,
        "contrast": "controller5303-controller5301",
        "N": 134,
        "wins": 27,
        "losses": 11,
        "ties": 96,
        "mean_paired_outcome_delta": 0.11940298507462686
      },
      {
        "next_wave": 2,
        "contrast": "controller5303-controller5302",
        "N": 134,
        "wins": 22,
        "losses": 11,
        "ties": 101,
        "mean_paired_outcome_delta": 0.08208955223880597
      }
    ],
    "wave3": [
      {
        "next_wave": 3,
        "contrast": "controller5303-controller5301",
        "N": 98,
        "wins": 10,
        "losses": 20,
        "ties": 68,
        "mean_paired_outcome_delta": -0.10204081632653061
      },
      {
        "next_wave": 3,
        "contrast": "controller5303-controller5302",
        "N": 98,
        "wins": 16,
        "losses": 11,
        "ties": 71,
        "mean_paired_outcome_delta": 0.05102040816326531
      }
    ]
  },
  "canonical": {
    "wave2": [
      {
        "next_wave": 2,
        "contrast": "controller5303-controller5301",
        "N": 134,
        "wins": 24,
        "losses": 11,
        "ties": 99,
        "mean_paired_outcome_delta": 0.09701492537313433
      },
      {
        "next_wave": 2,
        "contrast": "controller5303-controller5302",
        "N": 134,
        "wins": 27,
        "losses": 10,
        "ties": 97,
        "mean_paired_outcome_delta": 0.12686567164179105
      }
    ],
    "wave3": [
      {
        "next_wave": 3,
        "contrast": "controller5303-controller5301",
        "N": 98,
        "wins": 12,
        "losses": 16,
        "ties": 70,
        "mean_paired_outcome_delta": -0.04081632653061224
      },
      {
        "next_wave": 3,
        "contrast": "controller5303-controller5302",
        "N": 98,
        "wins": 23,
        "losses": 10,
        "ties": 65,
        "mean_paired_outcome_delta": 0.1326530612244898
      }
    ]
  }
}

## Native-vs-canonical consistency
CONSISTENT

## Ground risk and death precursor
{
  "classification": {
    "label": "GROUND_ASSOCIATED_NOT_PRIMARY",
    "criteria": {
      "A_overall_exposure": true,
      "B_matched_exposure": false,
      "C_death_precursor": true,
      "D_task_association": true
    },
    "overall_ground_risk_ratio": {
      "5301": 0.009392053331760125,
      "5302": 0.010000528329798555,
      "5303": 0.0019387937719978524
    },
    "matched_lower_contrasts": 1,
    "matched_contrast_count": 4
  },
  "matched": [
    {
      "wave": 1,
      "contrast": "5303-5301",
      "N": 36,
      "mean_difference": 0.0,
      "median_difference": 0.0,
      "direction_fraction_5303_lower": 0.0
    },
    {
      "wave": 1,
      "contrast": "5303-5302",
      "N": 36,
      "mean_difference": 0.0,
      "median_difference": 0.0,
      "direction_fraction_5303_lower": 0.0
    },
    {
      "wave": 2,
      "contrast": "5303-5301",
      "N": 16,
      "mean_difference": 0.0,
      "median_difference": 0.0,
      "direction_fraction_5303_lower": 0.0
    },
    {
      "wave": 2,
      "contrast": "5303-5302",
      "N": 16,
      "mean_difference": -0.004263425253991292,
      "median_difference": 0.0,
      "direction_fraction_5303_lower": 0.0625
    }
  ],
  "precursors": [
    {
      "policy_seed": 5301,
      "t_minus": 1,
      "death_event_N": 37,
      "available_N": 37,
      "median_altitude": 0.8034495123440459,
      "median_pitch": -0.2695931495577434,
      "median_vertical_speed": -42.90353202158333,
      "median_action_pitch": 0.0010404955828562379,
      "median_commanded_pitch": -0.2576309393891404,
      "median_time_to_ground": 0.03386110100409385,
      "guard_risk_fraction": 1.0
    },
    {
      "policy_seed": 5301,
      "t_minus": 5,
      "death_event_N": 37,
      "available_N": 37,
      "median_altitude": 21.250694642158198,
      "median_pitch": -0.27134835439261984,
      "median_vertical_speed": -43.09939589000192,
      "median_action_pitch": 0.0026115269865840673,
      "median_commanded_pitch": -0.2526448177132177,
      "median_time_to_ground": 0.42783656156874117,
      "guard_risk_fraction": 1.0
    },
    {
      "policy_seed": 5301,
      "t_minus": 10,
      "death_event_N": 37,
      "available_N": 37,
      "median_altitude": 42.877879087566775,
      "median_pitch": -0.27084412198134195,
      "median_vertical_speed": -43.415532344275874,
      "median_action_pitch": 0.0021334257908165455,
      "median_commanded_pitch": -0.24361264157789864,
      "median_time_to_ground": 0.9152860634229608,
      "guard_risk_fraction": 1.0
    },
    {
      "policy_seed": 5301,
      "t_minus": 25,
      "death_event_N": 37,
      "available_N": 37,
      "median_altitude": 108.74901291179413,
      "median_pitch": -0.27421469257614806,
      "median_vertical_speed": -44.39722158772239,
      "median_action_pitch": -0.00402800040319562,
      "median_commanded_pitch": -0.2652982468694422,
      "median_time_to_ground": 2.355653565209561,
      "guard_risk_fraction": 1.0
    },
    {
      "policy_seed": 5301,
      "t_minus": 50,
      "death_event_N": 37,
      "available_N": 37,
      "median_altitude": 220.92757915607476,
      "median_pitch": -0.2603187585753927,
      "median_vertical_speed": -45.07372079008915,
      "median_action_pitch": -6.91041350364685e-05,
      "median_commanded_pitch": -0.27462559971463685,
      "median_time_to_ground": 4.7005530501318145,
      "guard_risk_fraction": 0.02702702702702703
    },
    {
      "policy_seed": 5302,
      "t_minus": 1,
      "death_event_N": 28,
      "available_N": 28,
      "median_altitude": 3.481543874719426,
      "median_pitch": -0.40805998394979426,
      "median_vertical_speed": -65.94526176766006,
      "median_action_pitch": -0.017025819106493145,
      "median_commanded_pitch": -0.4284145036067525,
      "median_time_to_ground": 0.051197011568742914,
      "guard_risk_fraction": 1.0
    },
    {
      "policy_seed": 5302,
      "t_minus": 5,
      "death_event_N": 28,
      "available_N": 28,
      "median_altitude": 30.812089906963163,
      "median_pitch": -0.4042221091284143,
      "median_vertical_speed": -68.39434169365,
      "median_action_pitch": 0.0077407704666256905,
      "median_commanded_pitch": -0.4080118772482146,
      "median_time_to_ground": 0.429607031374744,
      "guard_risk_fraction": 1.0
    },
    {
      "policy_seed": 5302,
      "t_minus": 10,
      "death_event_N": 28,
      "available_N": 28,
      "median_altitude": 65.40484372978699,
      "median_pitch": -0.4030636301003486,
      "median_vertical_speed": -67.62259022358592,
      "median_action_pitch": -0.02643603435717523,
      "median_commanded_pitch": -0.40681044744250383,
      "median_time_to_ground": 0.889127958078425,
      "guard_risk_fraction": 1.0
    },
    {
      "policy_seed": 5302,
      "t_minus": 25,
      "death_event_N": 28,
      "available_N": 28,
      "median_altitude": 169.72886020787672,
      "median_pitch": -0.4266728858960487,
      "median_vertical_speed": -68.62871545250078,
      "median_action_pitch": 0.01155316992662847,
      "median_commanded_pitch": -0.3839859150110603,
      "median_time_to_ground": 2.3091192333339636,
      "guard_risk_fraction": 1.0
    },
    {
      "policy_seed": 5302,
      "t_minus": 50,
      "death_event_N": 28,
      "available_N": 28,
      "median_altitude": 338.5814050255812,
      "median_pitch": -0.4025183767693248,
      "median_vertical_speed": -66.20100441251854,
      "median_action_pitch": -0.036303685978055,
      "median_commanded_pitch": -0.4197998353424185,
      "median_time_to_ground": 4.601858594710615,
      "guard_risk_fraction": 0.2857142857142857
    },
    {
      "policy_seed": 5303,
      "t_minus": 1,
      "death_event_N": 5,
      "available_N": 5,
      "median_altitude": 0.8546051855208765,
      "median_pitch": -0.2620212432843033,
      "median_vertical_speed": -47.6734710526121,
      "median_action_pitch": 0.13009241223335266,
      "median_commanded_pitch": -0.05153998008929728,
      "median_time_to_ground": 0.0414692060500958,
      "guard_risk_fraction": 1.0
    },
    {
      "policy_seed": 5303,
      "t_minus": 5,
      "death_event_N": 5,
      "available_N": 5,
      "median_altitude": 19.581765610475905,
      "median_pitch": -0.2572808433696251,
      "median_vertical_speed": -46.86584610560514,
      "median_action_pitch": 0.1728707104921341,
      "median_commanded_pitch": -0.039712659147307705,
      "median_time_to_ground": 0.38203098232275623,
      "guard_risk_fraction": 1.0
    },
    {
      "policy_seed": 5303,
      "t_minus": 10,
      "death_event_N": 5,
      "available_N": 5,
      "median_altitude": 41.81605008553394,
      "median_pitch": -0.27147484393195526,
      "median_vertical_speed": -42.53557832052272,
      "median_action_pitch": 0.20107872784137726,
      "median_commanded_pitch": -0.06090569253873787,
      "median_time_to_ground": 0.7792110374731694,
      "guard_risk_fraction": 1.0
    },
    {
      "policy_seed": 5303,
      "t_minus": 25,
      "death_event_N": 5,
      "available_N": 5,
      "median_altitude": 114.46481391942585,
      "median_pitch": -0.4481780672842589,
      "median_vertical_speed": -66.37786342502739,
      "median_action_pitch": 0.0894286036491394,
      "median_commanded_pitch": -0.20776028022968807,
      "median_time_to_ground": 1.7386425699143642,
      "guard_risk_fraction": 1.0
    },
    {
      "policy_seed": 5303,
      "t_minus": 50,
      "death_event_N": 5,
      "available_N": 5,
      "median_altitude": 306.9060739149303,
      "median_pitch": -0.5559500938748834,
      "median_vertical_speed": -79.16258211813556,
      "median_action_pitch": 0.016829000785946846,
      "median_commanded_pitch": -0.361528904178015,
      "median_time_to_ground": 3.8769083284439807,
      "guard_risk_fraction": 0.6
    }
  ]
}

## Wave duration
[
  {
    "policy_seed": 5301,
    "wave": 1,
    "record_duration_mean": 408.9,
    "clear_duration_mean": 305.7906976744186,
    "record_N": 50,
    "clear_N": 43
  },
  {
    "policy_seed": 5301,
    "wave": 2,
    "record_duration_mean": 631.0697674418604,
    "clear_duration_mean": 276.6666666666667,
    "record_N": 43,
    "clear_N": 27
  },
  {
    "policy_seed": 5301,
    "wave": 3,
    "record_duration_mean": 391.6666666666667,
    "clear_duration_mean": 363.3,
    "record_N": 27,
    "clear_N": 20
  },
  {
    "policy_seed": 5302,
    "wave": 1,
    "record_duration_mean": 386.12,
    "clear_duration_mean": 349.7906976744186,
    "record_N": 50,
    "clear_N": 43
  },
  {
    "policy_seed": 5302,
    "wave": 2,
    "record_duration_mean": 338.90697674418607,
    "clear_duration_mean": 320.41379310344826,
    "record_N": 43,
    "clear_N": 29
  },
  {
    "policy_seed": 5302,
    "wave": 3,
    "record_duration_mean": 340.8965517241379,
    "clear_duration_mean": 396.2142857142857,
    "record_N": 29,
    "clear_N": 14
  },
  {
    "policy_seed": 5303,
    "wave": 1,
    "record_duration_mean": 347.84,
    "clear_duration_mean": 341.4791666666667,
    "record_N": 50,
    "clear_N": 48
  },
  {
    "policy_seed": 5303,
    "wave": 2,
    "record_duration_mean": 344.2708333333333,
    "clear_duration_mean": 324.95238095238096,
    "record_N": 48,
    "clear_N": 42
  },
  {
    "policy_seed": 5303,
    "wave": 3,
    "record_duration_mean": 262.6666666666667,
    "clear_duration_mean": 278.0689655172414,
    "record_N": 42,
    "clear_N": 29
  }
]

## Final mechanism classification
{
  "native_entry_wave2": "ENTRY_SUPPORTED_DIRECTIONALLY",
  "native_controller_wave2": "CONTROLLER_SUPPORTED_DIRECTIONALLY",
  "native_entry_wave3": "ENTRY_SUPPORTED_DIRECTIONALLY",
  "native_controller_wave3": "CONTROLLER_NOT_SUPPORTED",
  "canonical_entry_wave2": "ENTRY_SUPPORTED_STRONG",
  "canonical_controller_wave2": "CONTROLLER_SUPPORTED_DIRECTIONALLY",
  "canonical_entry_wave3": "ENTRY_SUPPORTED_DIRECTIONALLY",
  "canonical_controller_wave3": "CONTROLLER_MIXED_POSITIVE",
  "canonical_entry_overall": "ENTRY_OVERALL_SUPPORTED",
  "canonical_controller_overall": "CONTROLLER_OVERALL_MIXED",
  "final": "STATE_QUALITY_DOMINANT"
}

## Reward/credit interpretation
Mechanism labels are development diagnostics and do not by themselves establish a reward intervention.

## Next experiment
minimal inter-wave state-quality credit intervention

No training, no policy update, and no 45M use occurred. Matched future RNG is initial-stream matched, not event-wise coupled.
