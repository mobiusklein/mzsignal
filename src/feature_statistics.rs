//! Fitting methods for elution-over-time profile peak shape models.
//!
//! ![A skewed gaussian peak shape fit to an asymmetric profile][peak_fit]
//!
//! This covers multiple peak shape kinds and has some support
//! for multi-modal profiles.
//!
//! The supported peak shape types:
//! - [`GaussianPeakShape`]
//! - [`SkewedGaussianPeakShape`]
//! - [`BiGaussianPeakShape`]
//!
//! and the [`PeakShape`] type that can be used when dealing with
//!
//! Most of the fitting methods expect to work with [`PeakFitArgs`] which
//! can be created from borrowed signal-over-time data.
//!
//! # Example
//!
//! ```rust
//! # use mzsignal::text::load_feature_table;
//! use mzpeaks::feature::Feature;
//! use mzsignal::feature_statistics::{PeakFitArgs, SplittingPeakShapeFitter, FitConfig};
//!
//! # fn main() {
//! let features: Vec<Feature<_, _>> = load_feature_table("test/data/features_graph.txt").unwrap();
//! let feature = &features[10979];
//! let args = PeakFitArgs::from(feature);
//! let mut fitter = SplittingPeakShapeFitter::new(args);
//! fitter.fit_with(FitConfig::default().max_iter(10_000).smooth(1));
//! let z = fitter.score();
//! eprintln!("Score: {z}");
//! # }
//! ```
//!
//! # Model Fit Evaluation
//!
//! All peak shape models are optimized using a mean squared error (MSE) loss function, regularizing over the
//! position and shape parameters but not the amplitude parameter. Parameters are updated using a basic gradient
//! descent procedure.
//!
//! A peak shape fit isn't just about minimizing the residual error, it's about there actually being a peak, so
//! for downstream applications, we provide a [`PeakShapeModel::score`] method which compares the MSE of the model
//! to a straight line linear model of the form $`y = \alpha + \beta\times x`$. Prior work has shown this approach
//! can be more effective at distinguishing jagged noise regions where a peak shape can *fit*, but isn't meaningful.
//!
//!
//!
//! [peak_fit]: data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnQAAAEaCAYAAACYf8p4AAAAEXRFWHRncmFwaF9pbmRleAAxNDIxNp/ZP3oAAAB6dEVYdG1vZGVsX3BhcmFtcwB7Im11IjogMTIxLjUxODg3NjE2NDk2MTY1LCAic2lnbWEiOiAwLjE0MjA5Mjk5MjcyOTA4NDM1LCAiYW1wIjogNDY5MDc2OC41OTA0MTMwNDcsICJsYW0iOiAwLjMyMzMwMjAwOTI2NjI5NDZ90asA3wAAVL9JREFUeJzt3Xd4VGXawOHfmUnvvTd6CzVUaYogootiAcECKHYF3VXRVVnFVex%2BsCqKDRAbFkQEFAFBQKRIJwQIkJDeSC%2BTMnO%2BPyYZCCkkYZLJJM99XbmS8572nBwSnrxVUVVVRQghhBBCWC2NpQMQQgghhBCXRxI6IYQQQggrJwmdEEIIIYSVk4ROCCGEEMLKSUInhBBCCGHlJKETQgghhLByktAJIYQQQlg5SeiEEEIIIaycJHRCCCGEEFZOEjohhBBCCCsnCZ0QQgghhJWThE4IIYQQwsq1y4SusLCQF154gWuvvRYvLy8URWHZsmWXfd1NmzYxZswY3N3dcXV1JSoqipUrV15%2BwEIIIYQQ9WiXCV1WVhYvvfQSMTEx9O3b1yzXXLp0Kddccw22trYsWLCAN998k1GjRpGYmGiW6wshhBBC1MXG0gFYQmBgIKmpqQQEBPD3338zaNCgy7pefHw8jzzyCLNnz2bRokVmilIIIYQQomHaZQ2dvb09AQEBDTr2l19%2BYeTIkTg7O%2BPq6sr1119PdHR0tWM%2B/PBD9Ho9L730EmBs0lVV1exxCyGEEELUpl0mdA21YsUKrr/%2BelxcXHj99deZN28ex44dY8SIEcTHx5uO27RpE927d2f9%2BvWEhITg6uqKt7c38%2BbNw2AwWO4BhBBCCNEutMsm14YoLCxkzpw53HvvvXz00Uem8hkzZtCtWzcWLFhgKo%2BNjUWr1XL33Xczd%2B5c%2Bvbty6pVq3j55ZepqKjg1VdftdRjCCGEEKIdkISuDhs3biQ3N5dp06aRlZVlKtdqtQwZMoQtW7aYygoLCzEYDLz22ms8/fTTANxyyy1kZ2ezaNEinn32WVxdXVv8GYQQQgjRPkhCV4fY2FgAxowZU%2Bt%2BNzc309eOjo4UFRUxbdq0asdMmzaNX3/9lQMHDjBq1KjmC1YIIYQQ7ZokdHWo6vu2YsWKWgdQ2Nic/9YFBQURGxuLv79/tWP8/PwAyMnJacZIhRBCCNHeSUJXh06dOgHGpGzs2LH1HhsVFUVsbCzJycl07NjRVJ6SkgKAr69v8wUqhBBCiHZPRrnWYfz48bi5ubFgwQLKy8tr7M/MzDR9fdtttwHw6aefmsoMBgNLly7Fy8uLqKio5g9YCCGEEO1Wu62he%2B%2B998jNzTXVov38888kJSUBMHv2bNzd3fnggw%2B46667GDBgAFOnTsXX15eEhATWrVvH8OHDee%2B99wC48cYbufrqq3n11VfJysqib9%2B%2BrF69mh07drBkyRLs7e0t9pxCCCGEaPsUtZ3OgBsREcHZs2dr3RcXF0dERAQAW7du5bXXXmPXrl2UlpYSHBzMyJEjefTRR6vVvBUWFvL888%2BzcuVKsrOz6datG08//TR33HFHSzyOEEIIIdqxdpvQCSGEEEK0FdKHTgghhBDCyklCJ4QQQghh5SShE0IIIYSwcu0modPpdERHR6PT6SwdihBCCCGEWbWbhO706dNERkZy%2BvRpS4cihBBCCGFW7SahE0IIIYRoqyShE0IIIYSwcpLQCSGEEEJYOUnohBBCCCGsnCR0QgghhBBWThI6IYQQQggrJwmdldoem8mwVzfzwVaZhkUIIYRo7yShs1JL/4wnNU/HpzvOWDoUIYQQQliYJHRWSFVVDibmApBVWEZJmd6yAQkhhBDCoiShs0JJOSVkF5WZtpNzSywYjRBCCCEsTRI6K3SgsnauiiR0QgghRPsmCZ0VOpiQW207OUcSOiGEEKI9k4TOCh1Kyq22nZxbbJlAhBBCCNEqSEJnZcr1Bo4m51Urkxo6IYQQon2ThM7KHE8toLTCUK1M%2BtAJIYRob2bOnElERISlwzCJiIhg5syZFru/JHRW5uAFza19Qz0AqaETQggh2jsbSwcgGqdqQIS7oy3DO3lzKDGXtHwd5XoDtlrJz4UQQrQPH3/8MQaD4dIHthOSAViZg4k5gLF2LsTTCQCDCml5OkuGJYQQQrQoW1tb7O3tLR1GqyEJnRXJ15VzOrMIgH6hHgR7Opr2ST86IYQQbUlBQQGPP/44ERER2Nvb4%2Bfnx7hx49i/fz9Qex%2B6c%2BfOcdddd%2BHm5oaHhwczZszg0KFDKIrCsmXLTMfNnDkTFxcXkpOTmTRpEi4uLvj6%2BvLkk0%2Bi11dffemtt97iiiuuwNvbG0dHR6Kiovj%2B%2B%2B%2Bb%2B/EbTRI6K3I48fzo1n6h7gR7XJDQST86IYQQbciDDz7IBx98wC233MLixYt58skncXR0JCYmptbjDQYDEydO5Ouvv2bGjBm88sorpKamMmPGjFqP1%2Bv1jB8/Hm9vb9566y1Gjx7N22%2B/zUcffVTtuEWLFtG/f39eeuklFixYgI2NDZMnT2bdunVmf%2BbLIX3orEhVcytA3xAPnOzOvz6poRNCCNGWrFu3jvvuu4%2B3337bVDZ37tw6j1%2B9ejV//fUXCxcu5LHHHgPgoYceYty4cbUer9PpuO2225g3bx5gTCAHDBjAp59%2BykMPPWQ67uTJkzg6nq9AefTRRxkwYADvvPMO119//WU9ozlJQmdFDlbW0IV5OeHtYuw34O1sx7miMlIkoRNCCFFp/s/RHEvJt3QYAPQMcuOFib0afZ6Hhwe7d%2B8mJSWFoKCgSx7/66%2B/Ymtry3333Wcq02g0PPLII/z%2B%2B%2B%2B1nvPggw9W2x45ciQrVqyoVnZhMpeTk4Ner2fkyJF8/fXXjXmcZicJnZVQVZWDlWu4Vk1XAhDs6ci5ojKpoRNCCGFyLCWf3XHZlg7jsrzxxhvMmDGD0NBQoqKiuO6665g%2BfTodO3as9fizZ88SGBiIk5NTtfLOnTvXeryDgwO%2Bvr7Vyjw9PcnJyalWtnbtWl5%2B%2BWUOHjxIaWmpqVxRlKY8VrORhM5KJOeWkFVo/IfU78KEzsORw0l50odOCCGESc8gN0uHYNLUWKZMmcLIkSP58ccf%2Be2333jzzTd5/fXXWbVqFRMmTLjsuLRa7SWP2b59OzfccAOjRo1i8eLFBAYGYmtry9KlS/nqq68uOwZzkoTOShy6aEBElaqBEcm5Jaiq2ur%2BYhBCCNHymtLE2RoFBgby8MMP8/DDD5ORkcGAAQN45ZVXak3owsPD2bJlC8XFxdVq6U6dOtXk%2B//www84ODiwYcOGalOkLF26tMnXbC4yytVKVA2IsNEo9Aq6IKGrnLqktMJAVmGZRWITQgghzEmv15OXV33dcj8/P4KCgqo1e15o/PjxlJeX8/HHH5vKDAYD77//fpPj0Gq1KIpSbSqT%2BPh4Vq9e3eRrNhepobMSVf3negS64WB7vpq4k10OHhSQiyvJuSX4usoki0IIIaxbQUEBISEh3HrrrfTt2xcXFxc2bdrE3r17q416vdCkSZMYPHgwTzzxBKdOnaJ79%2B6sWbOG7GxjX8KmtGBdf/31vPPOO1x77bXcfvvtZGRk8P7779O5c2cOHz58Wc9oblJDZwUq9AaOJBv/Uulb1dxq0MPW1xjx6wSetPkWkLnohBBCtA1OTk48/PDDHDx4kBdeeIF//vOfnDhxgsWLF/Ovf/2r1nO0Wi3r1q3jtttuY/ny5Tz33HMEBQWZaugcHBwaHceYMWP49NNPSUtL4/HHH%2Bfrr7/m9ddf56abbrqs52sOiqqqqqWDaAnR0dFERkZy9OhRevWyrr4F0Sl5XP%2B/HQC8Nbkvt0aFgKrCipvgzBb0qsI/yhZw04Tx3D%2Bqk4WjFUIIIVqP1atXc9NNN7Fjxw6GDx9u6XCajdTQWYGq5laAfsEusOoBOLkBxi9AVbRoFZUXbZeTnF1suSCFEEIICyspqd5Spdfreffdd3Fzc2PAgAEWiqplSB86K3CoMqFzdbChY/Z2OPyN8WPShyiD74fdHzBEc5z9KRuA3haNVQghhLCU2bNnU1JSwrBhwygtLWXVqlXs3LmTBQsWVJsguC2ShM4KmCYUDvFAs%2B91Y6G9G/SYCN2upWDvl7ga8rk580MoexjsnOq%2BmBBCCNFGjRkzhrfffpu1a9ei0%2Bno3Lkz7777Lo8%2B%2BqilQ2t2ktC1coWlFcRmFAIw2rcIDmw27uhzG9i7APB70APcmPQm/mom7PwfXPmMpcIVQgghLOb222/n9ttvt3QYFiF96Fq5mNR8qoatjC1ed37HwHtMX6Z1nkK0IRwAdcdCyE1swQiFEEIIYWmS0LVyRyunK7GjnLCEVcbCsGHg39N0TJCnCy%2BWzwBAqSiBI9%2B2eJxCCCGEsBxJ6Fq5o8n5ANziuB9tSeVCywNnVTsm2NORvWp3PqyYyMERH8KI2ufoEUIIIUTbJAldKxedYqyhm2Fb2XfOyRt63lDtmJDK9Vxfq5jGIadhIOu5CiGEEO2KJHStmK5cT2xGIf5k06X8uLGw3x1gU315Lx8Xe%2By0xleZnCurRQghhBDtjSR0rdiJtAL0BpV0vNh07Ra4%2BoVqgyGqaDQKQR7GJU2Sc0pAXw5pR6AgvaVDFkIIIYQFSELXih2tbG4F6NqpE4z8F3h1qPXYYE9js2tBdgq8GgIfjoDoVS0SpxBCCCEsSxK6VqxqQISLvQ3hXvVPFhzkbkzojuXag51xfjpSDjRrfEIIIURbFB8fj6IoLFu2rNHnbt26FUVR2Lp1q9njqo8kdK3YsZQ83CikT4ADGk39Ax2qauiyisrQB/YzFkpCJ4QQQrQLktC1UuV6AzFpBTxus4oV6TfBR1dhmmG4FsEe59eoy/eMNH6RFQu6/OYOVQghhBAWJgldK3Uqo5CyCgM9lAS06EE11DsdSVUNHUCKc/fKr1RIO9zMkQohhBDC0iSha6WMK0So9NTEGwsCIus9PsTjfB%2B7WG2X8zuk2VUIIYQVevHFF1EUhZMnT3LnnXfi7u6Or68v8%2BbNQ1VVEhMTufHGG3FzcyMgIIC333672vkZGRnMmjULf39/HBwc6Nu3L8uXL69xn9zcXGbOnIm7uzseHh7MmDGD3NzcWmM6fvw4t956K15eXjg4ODBw4EDWrFnTHI/faJLQtVLRKfkEcQ53pdhY4N%2B73uMD3B1MFXinda7gGmjcSN7fjFEKIYQQzeu2227DYDDw2muvMWTIEF5%2B%2BWUWLlzIuHHjCA4O5vXXX6dz5848%2BeSTbNu2DYCSkhKuvPJKVqxYwR133MGbb76Ju7s7M2fOZNGiRaZrq6rKjTfeyIoVK7jzzjt5%2BeWXSUpKYsaMGTXiiI6OZujQocTExPDMM8/w9ttv4%2BzszKRJk/jxxx9b7PtRF6tK6F555RUURSEysv7aqrYgOiWPHpqz5wsuUUNnZ6PB3/WCueiC%2Bht3SA2dEEIIKzZ48GC%2B%2BuorHnroIX766SdCQkJ44oknuPvuu1m8eDEPPfQQa9euxdHRkc8%2B%2BwyAjz76iJiYGJYuXco777zD7Nmz2bx5M8OGDeP555%2BnoKAAgDVr1rBt2zZee%2B013n//fR599FF%2B%2B%2B033N3da8Tx2GOPERYWxv79%2B5k7dy6PPPIIW7duZdiwYTz99NMt%2Bj2pjY2lA2iopKQkFixYgLOzs6VDaXYGg0p0Sj73KAnnC/17XfK8YE9H0vJ1JOWWQPf%2BcGI9aLRQrgNbh2aMWAghRKtz4Es4%2BFX9xwT0hgmvnd9OPQy//vvS1757XfXtpdfXfly/26H/HZe%2BXj3uvfde09darZaBAweSlJTErFnn1zX38PCgW7dunDlzBoD169cTEBDAtGnTTMfY2toyZ84cpk2bxh9//ME//vEP1q9fj42NDQ899FC1e8yePZvt27ebyrKzs/n999956aWXKCgoMCWEAOPHj%2BeFF14gOTmZ4ODgy3rWy2E1Cd2TTz7J0KFD0ev1ZGVlWTqcZhV3rojiMj09bStr6NzDwNHzkucFeziy72yOsYZu0L0w%2BH5w9GjeYIUQQrROuQlwdkfjztHlNf4cqPuciBGNv9ZFwsLCqm27u7vj4OCAj49PjfJz584Zwzl7li5duqDRVG%2BI7NGjh2l/1efAwEBcXFyqHdetW7dq26dOnUJVVebNm8e8efNqjTMjI0MSukvZtm0b33//PQcOHGD27NmWDqfZRacYpxrpoVQmdJdobq1SNdI1LV9Hhb0HNlqralEXQghhTh5hEH6JhCrgov7ZDu6XPqc2dZ3jEVZ7eSNotdoGlYGxT1xzMBgMgLFyafz48bUe07lz52a5d0O1%2BoROr9cze/Zs7r33Xnr3rn9gQFsRnZyHA6WEKpnGAv8GJnSVc9HpDSrpBaXV5qYTQgjRzvS/o/HNnYF9ajanNkRTzmlG4eHhHD58GIPBUK2W7vjx46b9VZ83b95MYWFhtVq6EydOVLtex44dAWOz7dixY5s7/CZp9VU4H374IWfPnuW///1vg8/JyMggOjq62sepU6eaMUrzOpqShw57bvFYCfduNvZBaIAL56JLzikxflGcDXHb6zhDCCGEaHuuu%2B460tLSWLlypamsoqKCd999FxcXF0aPHm06rqKigg8%2B%2BMB0nF6v59133612PT8/P6688kqWLFlCampqjftlZmY205M0XKuuoTt37hz/%2Bc9/mDdvHr6%2Bvg0%2Bb/HixcyfP78ZI2s%2Bqqqa1nDtEuIPIX0bfG7IBTVyybnFsPt7%2BOUpY8FTp8HZp44zhRBCiLbj/vvvZ8mSJcycOZN9%2B/YRERHB999/z59//snChQtxdXUFYOLEiQwfPpxnnnmG%2BPh4evbsyapVq8jLy6txzffff58RI0bQu3dv7rvvPjp27Eh6ejp//fUXSUlJHDp0qKUfs5pWndA9//zzeHl5Nbrf3MMPP8zkyZOrlZ06dYpJkyaZMbrmkZxbQl5JOQCRQW6NOrdGDV34Be35KQehS%2BusJhZCCCHMydHRka1bt/LMM8%2BwfPly8vPz6datG0uXLmXmzJmm4zQaDWvWrOHxxx/niy%2B%2BQFEUbrjhBt5%2B%2B2369%2B9f7Zo9e/bk77//Zv78%2BSxbtoxz587h5%2BdH//79%2Bc9//tPCT1iTojZXD8LLFBsbS/fu3Vm4cCETJ040lU%2BdOpWcnBw2bNiAm5sbXl5eDbpedHQ0kZGRHD16lF69Lj0FiKX8ejSNB7/YhxtFfPrgWAZFNOz5qvR/6TdyisuZNjiUV68NgTc6GHdc9TyMfqoZIhZCCCGEpbXaPnTJyckYDAbmzJlDhw4dTB%2B7d%2B/m5MmTdOjQgZdeesnSYZpddEoeCgb%2Bsn%2BUqO%2BGwO4ljTq/qpYuKacEnLzAM8K4QyYYFkIIIdqsVtvkGhkZWetSGlUzPC9atIhOnTpZILLmdTQ5j3AlHWelFIoyQGvbqPODPRw5mpx/flBEUH/IiZeETgghhGjDWm1C5%2BPjU2uft4ULFwJYRX%2B4pohOySfqwhUiAvo06vzOfi5siE4n7lwR%2Bbpy3IL6Q/SPUJACBWngGmDmiIUQQghhaWZpcn399ddJTk42x6XatYx8HRkFpRes4aqAX49GXWNgZZ87VYV9Z3POr%2BkKUksnhBBCtFFmSeiee%2B45wsPDGTNmDEuXLq22xpm5bd26laNHjzbb9S2paoWInlUrRHh3ArvGrV0bFe6JRjF%2BvScuu3oNX3q0OcIUQgghRCtjloTu7NmzvPrqq2RnZzNr1iwCAgKYOnUq69atQ6/Xm%2BMW7cLRZOO8Nz00lU2uDVwh4kJuDrb0CDROd7I3Ltu4lquLP9g4QnmJuUIVQgghRCtiloQuODiYp556ioMHD3L48GHmzJnDrl27mDhxIoGBgcyePZvdu3eb41ZtWkJ2Me4UEqwYFxdu6BquFxvcwdjseigpF125Hh7cAc%2BmwNW1LygshBBCCOtm9mlLIiMjefXVV4mPj%2BePP/5g5MiRLF68mCuuuIKuXbvy8ssvk5GRYe7btglp%2BbrztXMA/k1bu3ZwZT%2B6cr3KwcRccPEDTaudoUYIIYQQl6lZ/pfX6XR88803vPHGG/z8889otVomTJhAZGQk//3vf%2BnUqVOtU5K0d6l5OrpXG%2BHatIRuUIfzkxHvicu%2B3LCEEEII0cqZLaFTVZXffvuNGTNm4O/vz%2B23305KSgpvvPEGSUlJrF27llWrVhEfH09UVBRPPPGEuW7dZqTl6Viuv4aFPVfCbV%2BCW1CTruPjYk9HX%2BNgir3xlQmdvgLOnYaiLHOFK4QQQohWwizz0P3zn/9k5cqVpKenExgYyIMPPsj06dNrXWIrMDCQe%2B%2B9l%2BnTp5vj1m1Gga6cwtIKQIO9XxfocXmTJg/p4MWZzCL2nc2hIj8dm0WRoC%2BDa1%2BHoQ%2BaJ2ghhBBCtApmSeg%2B/vhjbrrpJqZPn87YsWNRFKXe40eMGMHSpUvNces2Iz1fZ/o60N3hsq83KMKLr/ckUlymJzrXjr5ae2NCl3Xisq8thBBCiNbFLAldeno6zs4Nny8tIiKCiIgIc9y6zUjN02FHOXo0BJghoRt8YT%2B6%2BBz6%2BnaF5H2QefKyry2EEEKI1sUsfeh69%2B7NmjVr6ty/du1aOnbsaI5btVmpeTomaHZzzP4e%2Bv9yI%2BRd3sobIZ5OBFUmhrvjssGnm3FHliR0QgghRFtjloQuPj6ewsLCOvcXFhZy9uzZOvcL44CIHppE7JVy7M7FgLPPZV%2Bzqpbu77PZGLy7GAuLMqAk57KvLYQQQojWw2yjXOvrN7d37148PDzMdas2KS1fRyfFWCuneHcGG/vLvmbV9CW5xeWk2oad3yHNrkIIIUSb0uQ%2BdIsWLWLRokWAMZl7/PHHee6552ocl5eXR25uLrfffnvTo2wH0vJ0dFJSjBs%2BXcxyzSEX9KPbW%2BxLcNVG1gkIG2KWewghhBDC8pqc0Pn5%2BZmmJYmPjyc4OJjg4OBqxyiKgrOzM1FRUTz88MOXF2kbl5VbQLiSbtzw6WqWa3bydcHL2Y7sojK2pDsxSWMLhnLpRyeEEEK0MU1O6KZNm8a0adMAuOqqq3j%2B%2Bee5%2BuqrzRZYe2ObF4dWUY0bZkroFEVhUIQnG6LT2R2fj%2BrdCSXzOBSkmeX6QgghhGgdzDJtyZYtW8xxmXZLV67HtzQB7CoLzNTkCsb56DZEp5OWryN1yicEBYeAo6fZri%2BEEEIIy2tSQrdt2zYARo0aVW37UqqOF9Wl51/Qfw7A23wJ3ZAO3qavN2e5cVdnSeaEEEKItqZJCd2VV16JoiiUlJRgZ2dn2q6LqqooioJer29yoG1Zap6OzhrjCNdSR3/sHdzMdu2eQW4EujuQmqdj5d4E7hoabrZrCyGEEKJ1aFJCV9XEamdnV21bNE1ano755dP5suJq/jehI0FmvLZWo3DboFAWborlaHI%2Bh5Ny6eOrBUOFNL0KIYQQbUSTErrRo0fXuy0aJzVPRy6u/K12x7XvNWa//m2DQvnf5li0agVhywdBeQaMfgau%2BrfZ7yWEEEKIlme2iYVrc%2BbMGWJiYprzFm1Cer4OABd7G1wdbM1%2B/UB3R8Z096ccG4rKKkfSZp0w%2B32EEEIIYRlmSej%2B97//MXXq1Gpld999N126dCEyMpKBAweSkZFhjlu1Sal5JQAEVK692hzuGGpcKSLWUNmgK6tFCCGEEG2GWRK6Tz75BH9/f9P2hg0bWL58Offffz/vvvsuZ86cYf78%2Bea4VZvUP%2B17tts9xttlL0FpQbPcY1QXX4I9HDmlGhM69dwpMMggFSGEEKItMMs8dGfPnqVHjx6m7W%2B//ZYOHTrwwQcfAJCWlsaKFSvMcas2yaskjlBNJn5lhWDn0iz30GoUpg0O5fRmY0Kn6Esh9yx4dWyW%2BwkhhBCi5Zilhk5V1Wrbv/32GxMmTDBtR0REkJYmqxPUplxvIKgiEYAcx3CoZ/qXyzVlYChxhJwvkGZXIYQQok0wS0LXtWtXfvzxR8DY3JqSklItoUtKSsLDw8Mct2pzMgtK6aSkAlDi3qlZ7%2BXn5kBwl36m7ZKUY816PyGEEEK0DLMkdE8%2B%2BSQbN27E09OTiRMn0qNHD8aPH2/a//vvv9OvXz9z3KrNSc/KIlDJBsDgbZ41XOtz4xW9Oae6ApAYe7DZ7yeEEEKI5meWPnRTp07F29ub9evX4%2BHhwcMPP4yNjfHS2dnZeHl5cdddd5njVm1OUdL5aV3s/Ls3%2B/1GdPbhiDYUb8MxytNPmFbxEEIIIYT1MktCBzBu3DjGjRtXo9zLy4tVq1aZ6zZtjj7z/HxwbqE9m/1%2BGo3Cyf7P86%2BdySSo/nx1NodBEV7Nfl8hhBBCNJ9mnVhYXJpNdiwAelXBLaj5m1wBrrpqLAmaEMqx4evdCS1yTyGEEEI0H7ONcl2yZAmDBw/Gx8cHrVZb46OqCVZU51xwBoBUTQCKbfNNLHwhHxd7rurmB8BfZ861yD2FEEII0XzMkmXNnTuXd955h379%2BnHnnXfi6SmLvjfUZ/bT0Wf3p6ufI4%2B14H37hLjz27E0svPyySsux93J/EuOCSGEEKJlmCWhW758ObfccgvffvutOS7Xruwv9iHJMBSbgKAWve/tMY8ww/4wP%2BhHcjJjtPSjE0IIIayYWZpcS0pKGDt2rDku1a4YDCrp%2BTqgeddxrY0LRbgqJXRWkjme1jzLjQkhhBCiZZglobv66qvZu3evOS7VrpwrKqNcb1xlI9CtZRM62wDjiNoummROSkInhBBCWDWzJHSLFy9m165dLFiwgHPnpJN9Q5X99TE/2L3AGzZLCHRt2UEjip9xzjt/JZeklJQWvbcQQgghzMssCV23bt04c%2BYM8%2BbNw8/PD2dnZ9zc3Kp9uLu7m%2BNWbYqasp8oTSxjtAcI8HRt2Zv79jgfR0ZMjfV4hRBCCGE9zFItdMstt8hqA01gl3sagNNqEBEt3IcOv/OrUgSVnyU9v7TF%2B/EJIYQQwjzMktAtW7bMHJdpX1QV18o56OLUIAa62Lfs/T0i0Gsd0Op1dFGSOJFeIAmdEEIIYaVkpQhLKcrCUZ8PQIZdGFpNC9dwajQYfLoB0E1J5ERafsveXwghhBBmY7aELiEhgQcffJBu3brh6enJtm3bAMjKymLOnDkcOHDAXLdqG7JOmr7Md%2BlgkRDOj3RN4kRaoUViEEIIIcTlM0uT67Fjxxg5ciQGg4EhQ4Zw6tQpKioqAPDx8WHHjh0UFRXx6aefmuN2bcMFCV2ZZ2fLxDDkAV5O6stPyW4EpMvUJUIIIYS1MtvSXx4eHuzatQtFUfDz86u2//rrr2flypXmuFWboWadRAF0qi22XuGWCSKoPxXhdmQmx5OfXoDeoLZ8068QQgghLptZmly3bdvGQw89hK%2Bvb62jXcPCwkhOTjbHrdqMigxjDV2cGkiAh7PF4ugWYJwupbTCQEJ2scXiEEIIIUTTmaWGzmAw4OTkVOf%2BzMxM7O1beBRnK5fS%2B0GWnQikBHuGW3B0aVVCp2DgRFo%2BHXwsl1wKIYQQomnMUkM3YMAA1q1bV%2Bu%2BiooKvvnmG4YOHWqOW7UZcU59WaqfwDf6MQRaMKGLjPk/frf7F1/ZLpCBEUIIIYSVMktC9%2B9//5tff/2Vhx56iKNHjwKQnp7Opk2buOaaa4iJieGZZ54xx63ajLQ8nelr/xZex/VCdrpzdNSk0V2TwIm0PIvFIYQQQoimM0uT64QJE1i2bBmPPfYYH330EQB33nknqqri5ubG559/zqhRo8xxqzYjtZUkdFVLgHkqhWSkJgIDLReLEEIIIZrEbCvC33XXXdx8881s3LiR2NhYDAYDnTp1Yvz48bi6tvA6pa3dH28w7vA2SrXBfO84GTsbC87vfMESYI65sejK9TjYai0XjxBCCCEazSwJ3bZt2%2BjRowe%2Bvr5MmjSpxv6srCyOHTvWqFq6vXv3snz5crZs2UJ8fDze3t4MHTqUl19%2Bma5du5ojbMs5vYXIgp1UaDvxp/tdlo2lsoYOoBOJnM4spFeQuwUDEkIIIURjmaVq6KqrrmLjxo117t%2B8eTNXXXVVo675%2Buuv88MPP3D11VezaNEi7r//frZt28aAAQNM/fSskqpCejQAMYYwy6%2Bf6haE3s4NgK5KIifSZIJhIYQQwtqYpYZOVdV695eWlqLVNq4Z71//%2BhdfffUVdnZ2prLbbruN3r1789prr/HFF180KVaLy0uEUuPggxg1jABL9p8DUBQUv%2B6QtIeummQ2yooRQgghhNVpckKXkJBAfHy8afv48eOm9VsvlJuby5IlSwgPb9xqCFdccUWNsi5dutCrVy9iYmIaHW%2BrkXa%2BdvG4IYzRlq6hAzR%2BPYwJnZLEe6n5lg5HCCGEEI3U5IRu6dKlzJ8/H0VRUBSFV155hVdeeaXGcaqqotVqWbJkyWUFWnWt9PR0evXqddnXspjK5laA42oY073rnpC5xfgZ%2B9G5KcVkp50Fhlg2HiGEEEI0SpMTuilTphAZGYmqqkyZMoU5c%2BYwcuTIascoioKzszP9%2BvXD39//soP98ssvSU5O5qWXXqr3uIyMDDIzM6uVnTp16rLvbxbpxhq6JNWHfJwJ92oFKzN0uYbfzpSy6IgtJ3T25JWU4%2B5oa%2BmohBBCCNFATU7oevToQY8expqdpUuXMmrUKDp06GC2wC52/PhxHnnkEYYNG8aMGTPqPXbx4sXMnz%2B/2WK5LJUJXYwhDICw1lBD590J%2Bk4l%2BvA%2BAGLTCxgY4WXhoIQQQgjRUGYZFHGpBOtypaWlcf311%2BPu7s73339/yQEWDz/8MJMnT65WdurUqVqnVGlRZcVw7jRgbG71cLJtNTVhVWu6AhxPk4ROCCGEsCZmm1g4JiaGpUuXcubMGXJycmqMfFUUhc2bNzf6unl5eUyYMIHc3Fy2b99OUFDQJc/x8/PDz8%2Bv0fdqETct4eeNG9l%2BrjvhXq2gdq5SqKcTjrZaSsr1nJSRrkIIIYRVMUtCt2LFCu6%2B%2B25sbW3p1q0bnp6eNY651NQmtdHpdEycOJGTJ0%2ByadMmevbsaY5wLcfOCfrexmu/%2BJKsljDRuxX0n6ukObaKb%2B3/Dw8ljSdSP7d0OEIIIYRoBLMkdC%2B%2B%2BCL9%2B/fnl19%2BwcfHxxyXRK/Xc9ttt/HXX3/x008/MWzYMLNc19LKKgyk5pUAtKoaOoqz6V1xBDRQkB6Hqg5DURRLRyWEEEKIBjBLQpeSksKTTz5ptmQO4IknnmDNmjVMnDiR7OzsGhMJ33nnnWa7V0tKyinGUFlZ2SoGRFTxO78EmH9pHBkFpfhbetJjIYQQQjSIWRK6Pn36kJKSYo5LmRw8eBCAn3/%2BmZ9//rnGfqtL6FQVvrwVOyWAwUoH9qg9WlcN3QVrunZTkjiUmMs1vQIsGJAQQgghGsosa7m%2B8847fPrpp%2BzcudMclwNg69atqKpa54fVyUuCU5sIif2CHpoEAMJbUR86nL1RnXwB6KpJYlts5iVOEEIIIURrYZYautdffx13d3dGjhxJz549CQsLqzG1iKIo/PTTT%2Ba4nXVKP7/kV4whDHsbDX6u9hYMqCbFrzvEZ9JFSeL/TmSiqqr0oxNCCCGsgFkSusOHD6MoCmFhYRQWFnLs2LEax7T7xOCChO64GkqYlxMaTSv7nvj1hPjtdFGSSc4p4kxWEZ18XSwdlRBCCCEuwSwJXXx8vDku07alGRO6dMWXfFwY3JoGRFTx6w6Ao1JGqJLJ1hOZktAJIYQQVqBJCV1CgrEPWFhYWLXtS6k6vl1KjwYgWh8KQFhrWMP1Yn69TF9GKnFsPdGLWSOabzk3IYQQQphHkxK6iIgIFEWhpKQEOzs70/al6PX6ptzO%2BpUVQ7Zxya9jBmNCF94aa%2BgC%2B8CouSxN9OePGA/K47IpKdPjaFf/UmtCCCGEsKwmJXSfffYZiqJga2tbbVvUITMGVAMAMYZwoJXNQVfF1hHGPEdwdBpFMfugwsCuM%2Be4qnsrXUZNCCGEEEATE7qZM2fWuy0uklZ9QAS0slUiLnJFZx9stQrlepWtJzIkoRNCCCFaObPMQycuIXQwjJ1PtM8E4tRANAqEeLbehM7F3oaB4V4oGNh6UuajE0IIIVo7Sehagl8PGPE4H3o/jQENge6O2Nm00m%2B9vhxWP8wHOffxoHYtZ88VE59VZOmohBBCCFGPVppVtE0J54yJUascEFFFawvxO/AoSSBKcwKArScyLByUEEIIIeojCV0LOptdDLTyhA4gbCgAg7Sx0uwqhBBCWAFJ6Jpb/A747m50v7%2BJWpwDtNI56C4UOhgAdwrpqKSy68w5dOXtdMoZIYQQwgpIQtfc4rZD9Coctr1sKmr1NXShQ01fRmlOois3sDsu24IBCSGEEKI%2BktA1t7htABS4diIP4zJaYa14yhLAOIjD3g2AQZpYoPZ%2BdFmFpeTryls0NCGEEELUJAldcyothKQ9AJxxG2QqbvU1dBothAwE4Aq7UwD8ceJ8P7qjyXk8sOJvBr68iWve2UZGgc4iYQohhBDCqEkTC4sGOrsTDBUAHND2A8DL2Q5XB1sLBtVAoUPh9O8E65PwJJ8zWfDzoRRW7U9iywXJXVq%2Bjrc3nOT1W/tYMFghhBCifZMauuYU94fxs6Llj7IugBU0t1YJG2L6ckBls%2Bvsrw%2BYkjkbjUKwhyMA3%2B5L5GhyXsvHKIQQQghAErrmdWar8XNwFCdyjGvdtvrm1irBA%2BH6t1Ef3EG00/nkzk6r4Y4hYWx58kpWzBqMjUZBVWH%2Bz9GoqmrBgIUQQoj2S5pcm0thBqQb13CtiBhF6mljP7PWvIZrNfYuMOheFOCf4915f8tpru7hxwOjOhHg7mA6bOYVEXyyI4698TmsO5LKP/oEWS5mIYQQop2ShK65VI5uBUjzGYaqlgIQ5t3K56CrxW2DwrhtUFit%2B2Zf3YVVB5LJLirj1fXHGdvDHwdbbQtHKIQQQrRv0uTaXDqNgVs/g6i7ibXtZiq2mibXCxkMUF77SFZ3R1ueuKYrAMm5JXy87UxLRiaEEEIIJKFrPk5eEHkLTFxIfG6FqdhqmlwByopgxc3wegTs/aTOw6YOCqN7gCsAi7eeJi1PpjERQgghWpIkdC3g7DnjGq6Otlp8Xe0tHE0j2DpBRgyU5kHirjoP02oU/jOxJwAl5Xre%2BPV4S0UohBBCCCShax4GQ7XNhGxjQhfm5YSiKJaIqGkUxbSuK4l7oJ5RrFd08mF8L38AVh1IZm%2B8LBUmhBBCtBRJ6JrDD7Pgo6tgx0IAzp4rAiDMGvvPhVWu61qYDjnx9R763HU9sdMa/0nNWraXAwk5zRycEEIIIUASOvMz6OHMFkjZD2mHMRhUEnNKACvrP1elqoYOjLV09QjzdjI1vebrKrjzk93sPnOuOaMTQgghBJLQmV/aYSiprJnqMJq0fB1lFcYmWKsc4RrQB2yMK0KQ8NclD79zaDiv3twbRYGiMj0zlu5he2zmJc8TQgghRNNJQmduVatDAHS80jQgAqxzDjq0tuebXY%2BvBX35JU%2BZNjiMd6b0RatR0JUbmLXsbzYdS2/mQIUQQoj2SxI6cztTuX6rZwfwDCchu8i0yyqbXAH63Gb8XJQJsb816JSb%2Bofw3rT%2B2GoVyvQGHvxiH%2BsOpzZjkEIIIUT7JQmdOZXrzjdLdrwSgC3Hjc2NDrYagj0dLRTYZep5A9i5gsYGMhs%2BJcmE3oEsuSsKOxsNFQaVf357kIQLaiyFEEIIYR6S0JlT4m6oqJxUt%2BNoMgp0bIoxNjX%2Bo08Qtlor/XbbOcOU5fCv4zDyiUadOqa7Px9PHwhAWYWBBetjmiNCIYQQol2z0gyjlTL1n1MgYhTf70uiwmCcu23a4FCLhWUWna8GF98mnTq6qy839w8G4NfoNHaeyjJnZEIIIUS7JwmdOVU1twb2weDoxcq9iQB08XNhQJinBQOzvLnXdsfRVgvAS2uPUaE3XOKM8yr0BuatPsq9y/eSkS/LigkhhBAXk4TOnO5aDTN%2BhjH/YdeZc6YRrtMGh1nXChH1Kc6GPR9D2tFGnRbg7sAjV3UC4HhaAV9XJrsN8caGE6zYdZZNMRncv2IfunJ9o%2B4thBBCtHWS0JmTrQN0GAVdxpoSFjsbDTcPCLZwYGaiy4P/6wXrn4S/P2306feO7EhI5cCQd347QV7xpadA%2BflQCh9tO2PaPpiYy/Orj6LWswyZEEII0d5IQtcMsovK2HA0DYAJkQF4ONlZOCIzcXCH8OHGr4/8AOUljTvdVstz1/UAIKe4nIWbT9Z7/Im0AuZ%2BfxgANwcb%2BoS4A/D9viSW7YxvXOxCCCFEGyYJXTNYtT%2BJsso%2BYtMGh1k4GjPrf4fxc2keHF/X6NOvjQxgaEcvAD7/6yyx6QW1HpdXUs4DK/6mpFyPosD/pvXnkxkDCXBzAODldTH8KYMrhBBCCEASOrNTVZWv9yQA0NHHmSEdvCwckZl1uw4cKwd4HPii0acrisJ//tELjQJ6g8pLa4/VaD41GFQe/%2BYA8ZV9EJ8Y15Uru/nh5%2BrAR9OjsLfRoDeoPPLVfpnXTgghhEASOrPbG5/D6Uzj6hC3DQptO4MhqtjYQ%2B/Jxq/PbIXchg9uqNIzyI2plTWX22OzGPnGFh75cj8fbTvN7jPneHvjCbacME7IPK6nPw9f2dl0bp8QD167pTcAucXl3Pf53xSVVlzeMwkhhBBWzsbSAbQ131TWztlqFW6JCrFwNM2k3x2w5yNAhUPfwOinGn2JJ8Z1ZcPRNM4VlZGUU0JSTgnrjlRfGqyjrzPvTOmLRlM9Kb6pfwgxqQV8tO0MJ9ILuOK13xna0YsRnX24orMPHX2crSaRzijQcSwln8hgd3xc7Os8TlVVdpzK4nhqAUM7etO7sj%2BhEEIIAZLQmVVecbkpKbmmZ0C9/0FbtcC%2B4B8J6Ufh4Bcw8l%2Bg0TbqEt4u9qybM5If9idxKDGXQ0m5pOeXmvY722n56K4oXB1saz3/6Wu7czytgG0nM8krKWdDdDoboo2rcgS4OTCkoxe9g93pHexOr2B3XOxbzz91VVXZn5DD8p1n%2BeVoKuV6FVutwrWRgdw5JIzBHbxMCWlJmZ4fDySz9M84YjMKTdfoHezOHUPCuKFfEE52LfdsSTnFJOeUVItRCCGE5SlqO5n/ITo6msjISI4ePUqvXr2a5R7Ld8bzwppoAFbMGszILk1bWcEq7PoAfn3G%2BPX0n0xr116OtDwdh5JyOZVRyKguvpeshSrXG/jpYArbYzP589Q5sgpLaz1OUaCDjzN9gt0Z1dWXq7v74%2B5UM1E0GFT2xGfz4/5kYtLyGRThxR1Dwujo63LZzwagK9ez5mAKy/%2BKJzolv87jOvu5cPvgMM4VlfLV7gRy6pnexdXehkn9g7lnRAc6%2BDibJc66rD%2BSyj9XHqS0wsCwjt68enNvIpr5nqLhcovLKNBVEOrlZLZr5pWU893fifQJ8WBwW%2BsPLEQbIwmdmaiqyoRF2zmeVkColyN/PHlVjabCNqW0ED6/0bi2a/frLB0NqqoSm1HIjtgsdp7O4mBiLlmFZbUea6NRGNrRm/G9/LmmVwAFugp%2BPJDE6gMpJOfWnIplZBcf7hwaztXd/bDRalBVldQ8HcfT8jmeVoCtRsOYHn50qiPxO3uuiC92neXbv5PIKzmfnNlqFa7rHci4nv5sOpbO%2BiNpptHRFwtyd2DGFRFc1d2P9UdS%2BWZPImkXrJrhZKfl2weGERls/qZYVVVZvPU0b244Ua3c3kbD42O7ct/IDthctE5xbnEZ22KzKCmrYECYJ539XKRGr5mk5en48I/TfL0ngdIKA7cNDOWZCd3xdL686ZLKKgxM%2B3gX%2B87mAMYl/OZe241eQdLcf7HisgrK9SrujrW3KAjREiShM5OEc8WMfecPyvQGnhrfjUeu6nzpk6ydqhqrv1ohVVVJy9dxJCmPo8l5HEnOY39CbrWEqi42GoXOfi4cT6s%2BpUqguwOhnk4cT8snX1dzIEZXfxeu7RXA%2BMgAuge4seV4Bit2neWPk5nVjvN3s%2BeOIeFMHRyKn6uDqTy7qIzv9yXy5e4E0yojUeGe3DO8A%2BN7%2BVdLmir0Bn4/nsGXuxNM1/d3s2f1I8MJdHds%2BDfqEkor9Pz7hyOsOpAMGGsEx/Tw46eDKaZjegW58fotfbDVavj9eAa/H09n39kcDBf8ZvF2tmNwBy8Gd/BiSAdvegS6tqoELy6riJ8OJuNoqyXCx5kOPs6EeTnhYNu4rgQtKTm3hA%2B3nmbl3sQafwh4Odvx3HU9uHlAcJO/z8%2BvPsIXuxJqlN/YL4gnxnUjzNt8NYHWSlVVVu1P5sWfoyktNzDn6s48MLoTttqGjzfMLCjlrQ0n%2BPN0Fk9c05Wb%2BrfRvtei2UlCZ0Y5RWWsOpDMxD6B%2BLk5XPqEtqYkB%2BzdGt2frqWU6w3sictmQ3Qav0WnV6vhAugb6sEtA4L5R58gvJztOJVRwBe7EvhhXxIFjRxJ62irpeSiJcqGdvTizqHhjO8VUO8vfINB5UhyHg62WroFuF7yXp/tiOOltccA6BnoxncPDsO5jj6DCeeK8XC2xa2OvokXyi4q44EVf7M33lhDE%2BLpyGczB9HV35W98dk8/cNhzlSO6G6sUC9HJvYJ4sZ%2BwQ16xqZSVbXehKaotIL3tpzik%2B1nKNdX/1WoKBDk7kiolyMhnk4EezgS7OlIiIcjHX1dCHBv%2Bs%2B43qCibUANvqqqxJ8rJrOglJziMnKKysgpLudURiFrDiVXi3lEZx80GoVtF/wBMbSjFy9P6k1nv8Z1G/h2byJzfzBO6t03xJ3eIe58vScRfWWWbqtVuLl/CHcNC7/sWmGDQWVjTDrujrYMaWV9M%2Bv793OusJTnfjzKr9Fp1cp7Bbnx5q196RnkVu%2B1y/UGPv/rLAs3njT9flEU%2BN/U/kzsG2SeBxDtiiR0wjwS98B3dxsnHr7qWUtHc0kGg8qhpFy2nMjEVqNwXZ/AOptMi0or%2BOlgCj8eSKJMr9IjwJXuAa50D3Sje4ArWYVlbIhO49ejaRxJzqt2rou9DTcPCObOoeF09W%2BexEVVVV5YE83nf50F4Orufnw0fWC1hCEtT8cr62P4%2BVAKjrZapg4O5d6RHQn2qFmbV1xWwS9H0li4%2BSSJ2cYm6KhwT5bcFVVtoI%2BuXM/7W07xwdbTVFxQHdfR15kx3fwY08MPb2d79sSdY1dcNrvPZNfaz7Gbvys39AtiYp%2BgJtX6pOSWsCcum5jUfDILSsksLCWzoJSswjJyisuI8HZibE9/runpT79QT7QaBVVV%2BflwKgvWxdRI7BtqQJgHN/UP5vrKPwAulppXwv6zxj6hqXklpObpjJ9zdRSX67mmpz/z/tGToFreAcDxtHyeXXWE/Qm59cZxZTdfZo/pQlS4J6qqsv5IGvN/jiajwPi9ttUqjO7qR58Q4yChyGB3fF3rHrB1MDGXKR/%2BRZnegI%2BLHWseHUGQhyNxWUW8/dsJ1h6uPhp9QJgH04dFMKF3APY2jftjLq%2B4nH99e5DNxzMAYwL6zIQe9Av1aNR1Gny/knJ%2B3J9ER18XUwJcmy0nMnht/XHisooYGOHJ2B7%2BjO3hb/r3uTkmnad/OGL69%2BzjYo%2BHky2nKgcu2WgUHrmqM49c1Rk7m5p/vO08lcULa6KrDXSy0ShUGFRsNAofTY9iTHf/Jj%2Bn3qDy5oYT/HXmHNOHhnNT/%2BC23QVIAJLQCXMwGGDJSOOoVxS4axV0GmPpqCwiMbuYDdFpHEvJp3%2B4Jzf1D26REbYVegP3fv43Wyvn75t5RQQv3tCLcr2BpX/GsWhTLEVl1WsMbTQKN/YL5sHRHens58L%2BhBy%2B%2BzuJtYdTKbygRnJSvyBeu6VPnc2Px9PyWXMwBW8Xe8Z096tzcIaqqsRlFfHHyUzWHErhQC2JSmSwG9f1DuT63oGEe9e8jq5cT0J2MQcSctgdl82euGySchq%2BBJ23sx1juvuRmFPMrjPZpvJ%2BoR68dGMvQj2diDtXRHyW8eNMVhFJOSUk55aQWVD7oBsbjcLorr5c3yeQ3OJy9iXksP9sDql5l04UHW21PDa2C/cM72D6j7%2B4rIJFm2P5ZHucqUbsYvY2GkZ28WX2mM70rSX5ydeV885vJ1n%2BVzy1/YYPdHegX6gHI7v4MqqrDyGexkQls6CUie/uIC1fh41G4ct7hzCko3e1c48k5fG/32PZHJNeo1n9xn7BTOgdwIAwz0vWQB5NzuOhL/eZ/mi40HW9A3jymm6mAUnFZRWk5JaQnKujqLSCYR29G9VHUFVV1hxK4b9rY0xJWCdfZ2ZeEcHNA0JMNdpxWUX8d%2B0xfq9MMC/Wzd%2BVUC8nNsWkm8qu7RXAKzdF4uJgw7ubT/HBH6dN762bvyv9Qj0ordBTWmGgrMJATnFZtSS9g48z/5nYE3uthpnL9lJWYcDeRsOyuwczrJP3xSFcUoXewFPfH%2BbHym4SYGx9eGFiTwaEeTb6es2tqLSCVQeS%2BWFfEooCUwaGclP/4Fp/3yTlFPPJ9ji%2B35eEnY2GG/sFMTkq9JK1oe1Fq07oSktL%2Bc9//sOKFSvIycmhT58%2BvPzyy4wbN67R15KErpllHIePr4LyYnDygVm/gXcnS0fVrhSWVnDrBztNff/uHh7B9tgsU60BGP/zKSytYMdFy6YFujvUSEC8ne14YHRH7hvZsVmawRLOFfPz4RR%2BOpjMyfTCGvt7BroxupsvucVlxGcVc/ZcEan5uloTFDDWRPm5OuDjYoevqz0%2BLva4Otjw99kcDibm1nqet7MdT0/ozq0DQi5Zg6Er15OapyM5p4Sdp7P46WDtg2hq4%2B1sR6CHAwFujgR5OJBVWMr6I%2Beb6rr4ufDfSZGUlOl5fvVR03W1GoV7hkcwqqsvnk52eDrb4elki6OttkHv5HBSLp9sj%2BNQUq6pX2ZtOvo6M6qLL0eT8/i7chDEixN7MnN4hzrPScop5us9CXyzJ5FzRdUHIPm42HNNL3%2Bu7RVAVLhntS4Aqqqycm8i/1kTTVmFse/fNT39CfF04otdZ039AbUaha7%2BrqTlldQY6W2n1TA%2BMoBpg0IZ2tG73ncXn1XEvJ%2BOsj229qUCXR1smDooFI1G4bMdcaZmbFd7G8b18mf3mexa37OrvQ3zb%2BzFTf2r91M8mpzHk98dqtEH92JOdlpmj%2BnCPSMiTDWbG4%2Bl8%2BAX%2B9AbVJzttHx531BTbaW%2BsivGjthMMgtKGdPDn5EX1TKW6w3869tD/HwopbZbcnP/YOZe271J3QUy8nX8sD%2BZHw8kUVymZ1K/YKYPC29y96IzmYV8/tfZWru0eDrZcvuQMO4aGkGAuwPH0/JZ8scZ1hxKqfWPnF5BbtwaFcKN/YJrrS2voqoq%2B87m8MP%2BZOKzigjxdKSDrzMdvJ3p4OtMhLdzg/rNVugNnM0upqRM3ywD0ZqqVSd006ZN4/vvv%2Bfxxx%2BnS5cuLFu2jL1797JlyxZGjBjRqGtJQtcCDn4Nqx80fq21gyEPGkfBOnpYNKz2JCW3hEnv/2lqbqvS0deZ%2BTf0Mk2lczgplw//OM0vR9OqJTpajcJV3fyYMjCEq7r7Napz9%2BU4npbP%2BsOprDuSalpp5VJcHWwYFOHFkMrBFpHB7nXGm1Gg4/eYDDbFpLM9NosKg8r0YeE8PrZrk0cmGgwqe%2BOzWX0wmXWHU00DZXxc7IkK9yAq3JMBYZ70CnLH0a7mfxI7T2Ux76ejdT5vv1APFtzU22y1D3nF5RxNyeNwUh5HknPZdSab7KLaR4LfPCCYtyf3bVDSWFqh59ejaXy9J4E9cdnUVqno5mBDkIcjge4O6FVM/fy0GoWnr%2B1m%2BqMhMbuY/9t4kh8PJteZuF8s3NuJKQNDGdrRC1utBhuNBlutgo1Ww9pDKby75ZQpcfR2tuOZCd3JKyln2c74Wmt3FQWmRIXy5Phu%2BLrao6oqx9MK2HQsnU0x6RxKymNEZx9ev7VPrV0WwDhC%2BIOtp/luXyIVehV7Ww12Wo3pc/dAN%2BaM6VJrYvXTwWQeX3kQVQV3R1tmj%2BnMvrM57Dx9rsagrlAvR6YOCmPywBA8HO147JsD/HLU%2BIdCz0A3ltwVxdd7Evhke5wpUXa01dI72B1dhR5duR5duQFduR47Gw3d/F3pGeRGz0A3egS6EeThyNYTGXz7dyJbTmTWSKZstQoT%2BwRxz4gONRIbXbmetDwdWYWlZBcZuz9kF5WTU1xGTGp%2BjQQ72MMRVVVJueAPSxuNQq8gNw4lVe/KMqKzDyoqf546V61cq1GIDHJjYIQXgyI8iQr3wtfVnuTcElbtS%2BKH/UmmJSXrEujuQIS3MxE%2BToR7G5M8G43CyYwCTqYVcCK9kNMZhZTpDQyK8OS7B6%2Bo93otqdUmdHv27GHIkCG8%2BeabPPnkkwDodDoiIyPx8/Nj586djbqeJHQt5Ld5sPN/57cdveDKf8PAu0ErQ/pbwpGkPKYs%2BYuScj2OtlrmXN2FWSM61NqX50xmIZ/uiCMhu5hRXXyZ1D%2B43v5VLeFkegHrj6Sy/kgqJ9ML8XSyJczbmQhvJ8K9jL9kuwW40iPQrUEDCy6mK9ejUZRavx9NVVqh50hSHv5uDoR4Oja4RrOswsAnO87wv82x6MqN/%2BG62tswd0J3bh8c1qTnayiDQSU6JZ8/Tmaw7WQW%2BxJy0BtU%2BoS48%2B0Dw5o0wjersJRNx9L5NTqNP09l1RhociFfV3vem9a/RpMuwLGUfD7efoZzRWUEeziYBqQEuTtSrlf5bl8ivxxNMyVqDTFtcBjPXNvdNAel3qCyKSadpX/GmZrf%2B4d58OLEXrU2Y1fRleubffTzV7sTePbHI3Xur%2Bpvd%2BF2mLeTaZBSnxB3Pr9nMB5OxtqqhHPFLFgfU2MAx6VoFGok6F38XHCyt%2BFQYm618oHhnrg62JCWX1prrWpdRnT2YcYVEYzp7oeqqmyITuezP%2BNM0%2BVcGMt1vQN5cHQnU/KYlFPMqv3JfL8viYTs2hO1IHeHGjX7Wo1Cj0BXUnN1NWqXG8Pd0ZaD/xnXagbytNqEbu7cubzzzjtkZ2fj5nb%2BL9RXX32VZ599loSEBEJDQxt8PUnoWlDS37DhOUjcdb5s2jfQbYLlYmpnjibnsS02k0n9guvsdG8NSiv0je5ob42Scop5d/MptFqFx6/uYpFR8vm6ck6kFRBZR41iU663/WQW8eeKSMktIS1PR0qejswCHQPCPHl5UuRlPWducRmrDyTzzd7Eeps3u/m7suDmSKLC654Y%2BWR6AZkFpQy7RPNtS/po22kWrD8OGKehGd7Zh5GdfRjRxQdnextWH0jmq90JnEiv/uz9wzxYfs/gWkey7zydxWc74ijQVeBop8XBRouDrQYHWy0FpRXEpOYTl1VUo3bUxd6GiX2DmDIwhH6hHiiKwr6zOXz2Zxy/Hk2rs69nbexsNPi62HN1Dz%2BmDwuns1/tg8UOJuay9M84olPyGdLBi/tHday1Xy2cnxR%2Bc0w6f5/N4UhSXrWEt0qPwKqm2SDTAK98XTnxWUXEVX6cPVdc%2BbmoRlLq52pPtwBXuvq70s3fla4BrvQNcZeE7lLGjRtHcnIyx44dq1a%2BefNmxo4dy5o1a5g4cWKDrycJXQtTVYhZAxtfABd/uOfX83PW5SXD11MhqD/49QQnL3BwN0554uAODm5g4wjO3tWvpxoAxXidVvIDJISwrKom0cyCUsr1Bsr1auVnA24Otozu5ttiXQfM7WhyHooCPQLcak00jcsI5vLV7gR%2BPZpKVIQX79/ev84lExuiuKyCE2kFHEvNJz6riO4BbkzoHVDnEoPJuSV8vjOe346l42irJdDdAX93BwLdHAhwd8DX1R5vZ3s8nW3xcrZrcP/Py1FSpudQUi5/x2dzJDmPEE8nbh4Q3OhJsfOKy4k7V0S53kBnX5fLnqy7ubWeBS4vkpqaSmBgYI3yqrKUlNo7fQJkZGSQmVl9MtdTp06ZN0BRP0WBnjdC12uh%2BFz1BCzlAKQdNn7UxdETno4/v12SA2/U1kH7ouTuoZ3g1%2BP89uIrIPN4/bFOeB0G33d%2Be/1TsPfT%2Bs/peQNMXnZ%2BO3o1fH9P/ec4ecNTsee3i7Lgra71nwPw0J/yTCDPVEWeyajymRRFoUegGz3csuCtHvWfA1bxTFUiPcrrfU8KEAVEPfQnb00ebyxTlMt6Jiegf%2BWHydq6nyn4%2B3v4N/DvqrLc%2Bp%2BpJf7tOdppGdrRm6HHX4Mzle9pXx3n1POe3IF%2BtZ1z8TO1Eq02oSspKcHevmZfHgcHB9P%2BuixevJj58%2Bc3W2yiEWzswe2iSTJtHCB8BKQegrI6mko0F/3TNNQ1sa9Kvb2nVb3xozFUw6XPUS/uu6M24Jxa9jc2tqpz5Jnkmc4XyDPVV3YpbeSZqtV6tZFnqnFOa36mVqDVJnSOjo6Ultac90mn05n21%2BXhhx9m8uTJ1cpOnTrFpEmTzBqjaKIuY40fBgMUZUJpPujyzn%2BU5tdM6Gwd4cpnOZ/AXfD5Qk4XdbCOutt4j/oE9qu%2B3XksOHjUf47fRTUBPt1g5JP1n2PrWHP7UueAPFMVeSYjeSYjeabz5JmMLPVMrYT0oRNCCCGEsHKttqdov379OHnyJPn5%2BdXKd%2B/ebdovhBBCCCFacUJ36623otfr%2Beijj0xlpaWlLF26lCFDhjRqyhIhhBBCiLas1fahGzJkCJMnT%2Bbf//43GRkZdO7cmeXLlxMfH8%2Bnn15idJEQQgghRDvSahM6gM8//5x58%2BZVW8t17dq1jBo1ytKhCSGEEEK0Gq12UIS5yaAIIYQQQrRVrbqGzpyqpkCRCYaFEEIIYS06depkmoO3Pu0moUtMTASQueiEEEIIYTUa2rLYbppcc3Nz%2BeOPPwgNDa11BQrRcqomeV69ejWdO3e2dDgCeSetlbyX1kneS%2BvTlt%2BJ1NBdxMPDgxtvvNHSYYgLdO7cWfoztjLyTloneS%2Btk7yX1qc9v5NWOw%2BdEEIIIYRoGEnohBBCCCGsnCR0QgghhBBWThI60eJ8fX154YUX8PX1tXQoopK8k9ZJ3kvrJO%2Bl9ZF30o5GuQohhBBCtFVSQyeEEEIIYeUkoRNCCCGEsHKS0AkhhBBCWDlJ6IQQQgghrJwkdOKyFBYW8sILL3Dttdfi5eWFoigsW7as2jEGg4Fly5Zxww03EBoairOzM5GRkbz88svodLpqxyYmJjJ//nwGDx6Mp6cnPj4%2BXHnllWzatKkFn8q6mfudlJSUMGvWLCIjI3F3d8fFxYW%2BffuyaNEiysvLW/DJrJu538vFduzYgaIoKIpCVlZWMz5J29Ic76XqPVz88dprr7XQU1m35vpZSU9P54EHHiA4OBgHBwciIiKYNWtWCzxRy5BRruKyxMfH06FDB8LCwujYsSNbt25l6dKlzJw503RMYWEhrq6uDB06lH/84x/4%2Bfnx119/sXz5ckaNGsXvv/%2BOoigAvPfee8ydO5dJkyYxfPhwKioq%2BPzzz9m/fz%2BfffYZd999t4We1HqY%2B51kZ2dz3XXXMWrUKCIiItBoNOzcuZMvvviCqVOn8tVXX1noSa2Lud/LhQwGA1FRUcTGxlJUVERmZiY%2BPj4t%2BHTWqznei6IojBs3junTp1e7V//%2B/dvtslSN0RzvJDExkeHDhwNw3333ERwcTEpKCnv27GHNmjUt/YjNQxXiMuh0OjU1NVVVVVXdu3evCqhLly6tdkxpaan6559/1jh3/vz5KqBu3LjRVHb06FE1MzOzxj26d%2B%2BuhoSEmP8B2iBzv5O6PProoypgupeoX3O%2Blw8%2B%2BED19vZWH3vsMRWo8TMk6tYc7wVQH3nkkWaLua1rjncyYcIEtUOHDmpWVlazxW1p0uQqLou9vT0BAQH1HmNnZ8cVV1xRo/ymm24CICYmxlTWq1evGjUL9vb2XHfddSQlJVFQUGCGqNs2c7%2BTukRERACQm5vb6Bjbo%2BZ6L9nZ2Tz//PO89NJLeHh4mCXW9qQ5f15KSkou2VQuajL3Ozl%2B/Di//PILTz31FN7e3uh0ujbZXUQSOmExaWlpAA1qGkpLS8PJyQknJ6fmDqtdq%2B%2BdlJWVkZWVRWJiIj/%2B%2BCNvvfUW4eHhdO7cuaXDbHfqey/z5s0jICCABx54oKXDavfqey/Lli3D2dkZR0dHevbsKV0TWkht76SqD7a/vz9XX301jo6OODo6MmHCBOLj4y0RZrOwsXQAov164403cHNzY8KECfUed%2BrUKVatWsXkyZPRarUtFF37VN87WbVqFdOmTTNtDxw4kM8%2B%2BwwbG/k10tzqei%2BHDx9myZIlrF%2B/Xn42LKCu93LFFVcwZcoUOnToQEpKCu%2B//z533HEHeXl5PPTQQxaKtn2o7Z3ExsYCcP/99zNo0CBWrlxJQkIC8%2BfPZ%2BzYsRw%2BfLhNVBbIb2JhEQsWLGDTpk0sXry43mai4uJiJk%2BejKOjo4wQa2aXeidXXXUVGzduJDc3l82bN3Po0CGKiopaPtB2pr73MmfOHCZMmMA111xjmeDasfrey59//llt%2B5577iEqKopnn32WmTNn4ujo2IKRth91vZPCwkIAAgICWLduHRqNsXEyJCSEadOm8dVXX3HvvfdaImSzkiZX0eJWrlzJ888/z6xZs%2Br9a1Wv1zN16lSOHTvG999/T1BQUAtG2b405J34%2B/szduxYbr31Vj744AP%2B8Y9/MG7cOFMThzC/%2Bt7LypUr2blzJ2%2B//baFomu/Gvo7rIqdnR2PPvooubm57Nu3rwUibH/qeydVCfSUKVNMyRzA5MmTsbGxYefOnS0aa3ORhE60qI0bNzJ9%2BnSuv/56Pvzww3qPve%2B%2B%2B1i7di3Lli1jzJgxLRRh%2B9OYd3KhW2%2B9lcLCQn766admjK79utR7eeqpp5g8eTJ2dnbEx8cTHx9vGqCSmJhISkpKC0fcPjT15yU0NBQwDmIR5nWpd1JVGeDv71%2BtXKvV4u3tTU5OTovE2dykyVW0mN27d3PTTTcxcOBAvv3223r7Xj311FMsXbqUhQsXVuu3JcyrMe/kYiUlJQDk5eU1V3jtVkPeS2JiIl999VWtne0HDBhA3759OXjwYAtE235czs/LmTNnAPD19W2u8NqlhryTqKgoAJKTk6uVVw30aivvRBI60SJiYmK4/vrriYiIYO3atfX2IXnzzTd56623ePbZZ3nsscdaMMr2paHvJCsrC29v7xoT2n7yySeAcXCEMJ%2BGvpcff/yxRtk333zDypUr%2BfzzzwkJCWnuUNuVhr6XzMzMGglCQUEBCxcuxMfHx5RciMvX0Hdy5ZVX4ufnx5dffsmzzz6Lg4MDYByJrNfrGTduXEuG3WwkoROX7b333iM3N9fUxPPzzz%2BTlJQEwOzZs9FoNIwfP56cnByeeuop1q1bV%2B38Tp06MWzYMMD4n9TcuXPp0qULPXr04Isvvqh27Lhx42pUm4uazPlOvvjiCz788EMmTZpEx44dKSgoYMOGDWzcuJGJEydKc3gjmPO9TJo0qcb1q2rkJkyYICtFNII538v777/P6tWrmThxImFhYaSmpvLZZ5%2BRkJDAihUrsLOza9mHs1LmfCf29va8%2BeabzJgxg1GjRnHXXXeRkJDAokWLGDlyJDfffHPLPlxzsfTMxsL6hYeHq0CtH3FxcWpcXFyd%2BwF1xowZpmu98MIL9R67ZcsWiz2nNTHnO9m7d686efJkNSwsTLW3t1ednZ3VAQMGqO%2B8845aXl5uuYe0QuZ8L7Wp%2BvmRlSIax5zv5bffflPHjRunBgQEqLa2tqqHh4d6zTXXqJs3b7bcA1qh5vhZ%2Bfrrr9W%2Bffuq9vb2qr%2B/v/roo4%2Bq%2Bfn5Lf9wzUTWchVCCCGEsHIyylUIIYQQwspJQieEEEIIYeUkoRNCCCGEsHKS0AkhhBBCWDlJ6IQQQgghrJwkdEIIIYQQVk4SOiGEEEIIKycJnRBCCCGElZOETgghhBDCyklCJ4QQQghh5SShE0KIWkRERDBz5kxLhyGEEA0iCZ0Qol3buXMnL774Irm5uZYORQghmkxRVVW1dBBCCGEpb731Fk899RRxcXFERESYyktLS9FoNNja2louOCGEaCAbSwcghBCtkb29vaVDEEKIBpMmVyFEu/Xiiy/y1FNPAdChQwcURUFRFOLj42v0oVu2bBmKorBjxw7mzJmDr68vHh4ePPDAA5SVlZGbm8v06dPx9PTE09OTuXPncnEDiMFgYOHChfTq1QsHBwf8/f154IEHyMnJacnHFkK0QVJDJ4Rot26%2B%2BWZOnjzJ119/zf/93//h4%2BMDgK%2Bvb53nzJ49m4CAAObPn8%2BuXbv46KOP8PDwYOfOnYSFhbFgwQLWr1/Pm2%2B%2BSWRkJNOnTzed%2B8ADD7Bs2TLuvvtu5syZQ1xcHO%2B99x4HDhzgzz//lOZdIUTTqUII0Y69%2BeabKqDGxcVVKw8PD1dnzJhh2l66dKkKqOPHj1cNBoOpfNiwYaqiKOqDDz5oKquoqFBDQkLU0aNHm8q2b9%2BuAuqXX35Z7T6//vprreVCCNEY0uQqhBCNMGvWLBRFMW0PGTIEVVWZNWuWqUyr1TJw4EDOnDljKvvuu%2B9wd3dn3LhxZGVlmT6ioqJwcXFhy5YtLfocQoi2RZpchRCiEcLCwqptu7u7AxAaGlqj/MK%2BcbGxseTl5eHn51frdTMyMswcqRCiPZGETgghGkGr1Ta4XL1gUITBYMDPz48vv/yy1vPr67cnhBCXIgmdEKJdu7D5tDl16tSJTZs2MXz4cBwdHVvknkKI9kP60Akh2jVnZ2eAZl8pYsqUKej1ev773//W2FdRUSErVQghLovU0Akh2rWoqCgAnnvuOaZOnYqtrS0TJ040%2B31Gjx7NAw88wKuvvsrBgwe55pprsLW1JTY2lu%2B%2B%2B45FixZx6623mv2%2BQoj2QRI6IUS7NmjQIP773//y4Ycf8uuvv2IwGIiLi2uWe3344YdERUWxZMkSnn32WWxsbIiIiODOO%2B9k%2BPDhzXJPIUT7IGu5CiGEEEJYOelDJ4QQQghh5SShE0IIIYSwcpLQCSGEEEJYOUnohBBCCCGsnCR0QgghhBBWThI6IYQQQggrJwmdEEIIIYSVk4ROCCGEEMLKSUInhBBCCGHlJKETQgghhLByktAJIYQQQlg5SeiEEEIIIaycJHRCCCGEEFZOEjohhBBCCCsnCZ0QQgghhJX7f/I2QpG9gn2RAAAAAElFTkSuQmCC
use std::{
    borrow::Cow,
    f64::consts::{PI, SQRT_2},
    fmt::Debug,
    iter::FusedIterator,
    ops::{Deref, Range},
};

use libm::erf;

use mzpeaks::prelude::TimeArray;

use crate::arrayops::{trapz, ArrayPair, ArrayPairSplit};

/// An iterator over [`PeakFitArgs`] which explicitly casts the signal magnitude
/// from `f32` to `f64`.
///
/// This conversion is done specifically for convenience during model fitting.
pub struct PeakFitArgsIter<'a> {
    inner: std::iter::Zip<
        std::iter::Copied<std::slice::Iter<'a, f64>>,
        std::iter::Copied<std::slice::Iter<'a, f32>>,
    >,
}

impl<'a> Iterator for PeakFitArgsIter<'a> {
    type Item = (f64, f64);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(x, y)| (x, y as f64))
    }
}

impl<'a> FusedIterator for PeakFitArgsIter<'a> {}

impl<'a> ExactSizeIterator for PeakFitArgsIter<'a> {
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<'a> PeakFitArgsIter<'a> {
    pub fn new(
        inner: std::iter::Zip<
            std::iter::Copied<std::slice::Iter<'a, f64>>,
            std::iter::Copied<std::slice::Iter<'a, f32>>,
        >,
    ) -> Self {
        Self { inner }
    }
}

/// A point along a [`PeakFitArgs`] which produces the greatest
/// valley between two peaks.
///
/// Produced by [`PeakFitArgs::locate_extrema`]
#[derive(Debug, Default, Clone, Copy, PartialEq, PartialOrd)]
pub struct SplittingPoint {
    /// The signal magnitude of the first maximum point
    pub first_maximum_height: f32,
    /// The signal magnitude at the nadir of the valley
    pub minimum_height: f32,
    /// The signal magnitude of the second maximum point
    pub second_maximum_height: f32,
    /// The time coordinate of the nadir of the valley
    pub minimum_time: f64,
}

impl SplittingPoint {
    pub fn new(first_maximum: f32, minimum: f32, second_maximum: f32, minimum_index: f64) -> Self {
        Self {
            first_maximum_height: first_maximum,
            minimum_height: minimum,
            second_maximum_height: second_maximum,
            minimum_time: minimum_index,
        }
    }

    pub fn total_distance(&self) -> f32 {
        (self.first_maximum_height - self.minimum_height)
            + (self.second_maximum_height - self.minimum_height)
    }
}

/// Represent an array pair for signal-over-time data
#[derive(Debug, Default, Clone)]
pub struct PeakFitArgs<'a, 'b> {
    /// The time axis of the signal
    pub time: Cow<'a, [f64]>,
    /// The paired signal intensity to fit against
    pub intensity: Cow<'b, [f32]>,
}

impl<'c, 'd, 'a: 'c, 'b: 'd, 'e: 'c + 'd + 'a + 'b> PeakFitArgs<'a, 'b> {
    pub fn new(time: Cow<'a, [f64]>, intensity: Cow<'b, [f32]>) -> Self {
        assert_eq!(
            time.len(),
            intensity.len(),
            "time array length ({}) must equal intensity length ({})",
            time.len(),
            intensity.len()
        );
        Self { time, intensity }
    }

    /// Apply [`moving_average_dyn`](crate::smooth::moving_average_dyn) to the signal intensity
    /// returning a new [`PeakFitArgs`].
    pub fn smooth(&'e self, window_size: usize) -> PeakFitArgs<'a, 'd> {
        let mut store = self.borrow();
        let sink = store.intensity.to_mut();
        crate::smooth::moving_average_dyn(&self.intensity, sink.as_mut_slice(), window_size * 3);
        store
    }

    /// Find the indices of local maxima, optionally above some `min_height` threshold.
    pub fn peak_indices(&self, min_height: Option<f64>) -> Vec<usize> {
        let min_height = min_height.unwrap_or_default() as f32;
        let n = self.len();
        let n1 = n.saturating_sub(1);
        let mut indices = Vec::new();
        for (i, y) in self
            .intensity
            .iter()
            .copied()
            .enumerate()
            .take(n.saturating_sub(1))
            .skip(1)
        {
            if y > min_height
                && y >= self.intensity[i.saturating_sub(1)]
                && y >= self.intensity[(i + 1).min(n1)]
            {
                indices.push(i);
            }
        }
        indices
    }

    /// Find the indices of local minima, optionally below some `max_height` threshold.
    pub fn valley_indices(&self, max_height: Option<f64>) -> Vec<usize> {
        let max_height = max_height.unwrap_or(f64::INFINITY) as f32;
        let n = self.len();
        let n1 = n.saturating_sub(1);
        let mut indices = Vec::new();
        for (i, y) in self
            .intensity
            .iter()
            .copied()
            .enumerate()
            .take(n.saturating_sub(1))
            .skip(1)
        {
            if y < max_height
                && y <= self.intensity[i.saturating_sub(1)]
                && y <= self.intensity[(i + 1).min(n1)]
            {
                indices.push(i);
            }
        }
        indices
    }

    /// Find a point between two local maxima that would separate the signal between the two peaks
    pub fn locate_extrema(&self, min_height: Option<f64>) -> Option<SplittingPoint> {
        let maxima_indices = self.peak_indices(min_height);
        let minima_indices = self.valley_indices(None);

        let mut candidates = Vec::new();

        for (i, max_i) in maxima_indices.iter().copied().enumerate() {
            for j in (i + 1)..maxima_indices.len() {
                let max_j = maxima_indices[j];
                for min_k in minima_indices.iter().copied() {
                    if self.time[max_i] > self.time[min_k] || self.time[min_k] > self.time[max_j] {
                        continue;
                    }
                    let y_i = self.intensity[max_i];
                    let y_j = self.intensity[max_j];
                    let y_k = self.intensity[min_k];
                    if max_i < min_k
                        && min_k < max_j
                        && (y_i - y_k) > (y_i * 0.01)
                        && (y_j - y_k) > (y_j * 0.01)
                    {
                        candidates.push(SplittingPoint::new(y_i, y_k, y_j, self.time[min_k]));
                    }
                }
            }
        }
        let split_point = candidates
            .into_iter()
            .max_by(|a, b| a.total_distance().total_cmp(&b.total_distance()));
        split_point
    }

    /// Find the indices that separate the signal into discrete segments according
    /// to `split_points`.
    ///
    /// The returned [`Range`] can be extracted using [`PeakFitArgs::slice`]
    pub fn split_at(&self, split_points: &[SplittingPoint]) -> Vec<Range<usize>> {
        let n = self.len();
        let mut segments = Vec::new();
        let mut last_x = self.time.first().copied().unwrap_or_default() - 1.0;
        for point in split_points {
            let start_i = self
                .time
                .iter()
                .position(|t| *t > last_x && *t <= point.minimum_time)
                .unwrap_or_default();
            let end_i = self
                .time
                .iter()
                .rposition(|t| *t > last_x && *t <= point.minimum_time)
                .unwrap_or_default();
            if start_i != end_i {
                segments.push(start_i..(end_i + 1).min(n));
            }
            last_x = point.minimum_time;
        }

        let i = self.time.iter().position(|t| *t > last_x).unwrap_or(n);
        if i != n {
            segments.push(i..n);
        }
        segments
    }

    pub fn get(&self, index: usize) -> (f64, f32) {
        (self.time[index], self.intensity[index])
    }

    /// Select a sub-region of the signal given by `iv` like those returned by [`PeakFitArgs::split_at`]
    ///
    /// The returned instance will borrow the data from `self`
    pub fn slice(&'e self, iv: Range<usize>) -> PeakFitArgs<'c, 'd> {
        let x = &self.time[iv.clone()];
        let y = &self.intensity[iv.clone()];
        (x, y).into()
    }

    pub fn subtract(&mut self, intensities: &[f64]) {
        assert_eq!(self.intensity.len(), intensities.len());

        self.intensity
            .to_mut()
            .iter_mut()
            .zip(intensities.iter())
            .for_each(|(a, b)| {
                *a -= (*b) as f32;
            });
    }

    pub fn subtract_region(&mut self, time: Cow<'a, [f64]>, intensities: &[f64]) {
        let t_first = *time.first().unwrap();
        let i_first = self.find_time(t_first);
        self.intensity
            .to_mut()
            .iter_mut()
            .skip(i_first)
            .zip(intensities.iter())
            .for_each(|(a, b)| {
                *a -= (*b) as f32;
            });
    }

    /// Find the index nearest to `time`
    pub fn find_time(&self, time: f64) -> usize {
        let time_array = &self.time;
        let n = time_array.len().saturating_sub(1);
        let mut j = match time_array.binary_search_by(|x| x.partial_cmp(&time).unwrap()) {
            Ok(i) => i.min(n),
            Err(i) => i.min(n),
        };

        let i = j;
        let mut best = j;
        let err = (time_array[j] - time).abs();
        let mut best_err = err;
        let n = n + 1;
        // search backwards
        while j > 0 && j < n {
            let err = (time_array[j] - time).abs();
            if err < best_err {
                best_err = err;
                best = j;
            } else if err > best_err {
                break;
            }
            j -= 1;
        }
        j = i;
        // search forwards
        while j < n {
            let err = (time_array[j] - time).abs();
            if err < best_err {
                best_err = err;
                best = j;
            } else if err > best_err {
                break;
            }
            j += 1;
        }
        best
    }

    /// Integrate the area under the signal for this data using trapezoid integration
    pub fn integrate(&self) -> f32 {
        trapz(&self.time, &self.intensity)
    }

    /// Compute the mean over [`Self::time`] weighted by [`Self::intensity`]
    pub fn weighted_mean_time(&self) -> f64 {
        self.iter()
            .map(|(x, y)| ((x * y), y))
            .reduce(|(xa, ya), (x, y)| ((xa + x), (ya + y)))
            .map(|(x, y)| x / y)
            .unwrap_or_default()
    }

    /// Find the index where [`Self::intensity`] achieves its maximum value
    pub fn argmax(&self) -> usize {
        let mut ymax = 0.0;
        let mut ymax_i = 0;
        for (i, (_, y)) in self.iter().enumerate() {
            if y > ymax {
                ymax = y;
                ymax_i = i;
            }
        }
        ymax_i
    }

    /// The length of the arrays
    pub fn len(&self) -> usize {
        self.time.len()
    }

    /// Create a new [`PeakFitArgs`] from this one that borrows its data from this one
    pub fn borrow(&'e self) -> PeakFitArgs<'c, 'd> {
        let is_time_owned = matches!(self.time, Cow::Owned(_));
        let is_intensity_owned = matches!(self.intensity, Cow::Owned(_));
        let time = if is_time_owned {
            Cow::Borrowed(self.time.deref())
        } else {
            self.time.clone()
        };

        let intensity = if is_intensity_owned {
            Cow::Borrowed(self.intensity.deref())
        } else {
            self.intensity.clone()
        };
        Self::new(time, intensity)
    }

    /// Compute the "null model" residuals $`\sum_i(y_i - \bar{y})^2`$
    pub fn null_residuals(&self) -> f64 {
        let mean = self.intensity.iter().sum::<f32>() as f64 / self.len() as f64;
        self.intensity
            .iter()
            .map(|y| (*y as f64 - mean).powi(2))
            .sum()
    }

    /// Compute the simple linear model residuals $`\sum_i{(y_i - (\beta_{1}x_{i} + \beta_0))^2}`$
    pub fn linear_residuals(&self) -> f64 {
        let (xsum, ysum) = self
            .iter()
            .reduce(|(xa, ya), (x, y)| (xa + x, ya + y))
            .unwrap_or_default();

        let xmean = xsum / self.len() as f64;
        let ymean = ysum / self.len() as f64;

        let mut tss = 0.0;
        let mut hat = 0.0;

        for (x, y) in self.iter() {
            let delta_x = x - xmean;
            tss += delta_x.powi(2);
            hat += delta_x * (y - ymean);
        }

        let beta = hat / tss;
        let alpha = ymean - beta * xmean;

        self.iter()
            .map(|(x, y)| (y - ((x * beta) + alpha)).powi(2))
            .sum()
    }

    /// Create a [`PeakFitArgsIter`] which is a copying iterator that casts the intensity
    /// value to `f64` for convenience in model fitting.
    pub fn iter(&self) -> PeakFitArgsIter<'_> {
        PeakFitArgsIter::new(
            self.time
                .iter()
                .copied()
                .zip(self.intensity.iter().copied()),
        )
    }

    /// Borrow this data as an [`ArrayPairSplit`]
    pub fn as_array_pair(&'e self) -> ArrayPairSplit<'a, 'b> {
        let this = self.borrow();
        ArrayPairSplit::new(this.time, this.intensity)
    }
}

impl<'a, 'b> From<(Cow<'a, [f64]>, Cow<'b, [f32]>)> for PeakFitArgs<'a, 'b> {
    fn from(pair: (Cow<'a, [f64]>, Cow<'b, [f32]>)) -> PeakFitArgs<'a, 'b> {
        PeakFitArgs::new(pair.0, pair.1)
    }
}

impl<'a, 'b> From<(&'a [f64], &'b [f32])> for PeakFitArgs<'a, 'b> {
    fn from(pair: (&'a [f64], &'b [f32])) -> PeakFitArgs<'a, 'b> {
        PeakFitArgs::new(Cow::Borrowed(pair.0), Cow::Borrowed(pair.1))
    }
}

impl From<(Vec<f64>, Vec<f32>)> for PeakFitArgs<'static, 'static> {
    fn from(pair: (Vec<f64>, Vec<f32>)) -> PeakFitArgs<'static, 'static> {
        let mz_array = Cow::Owned(pair.0);
        let intensity_array = Cow::Owned(pair.1);
        PeakFitArgs::new(mz_array, intensity_array)
    }
}

impl<'a, 'b> From<PeakFitArgs<'a, 'b>> for ArrayPairSplit<'a, 'b> {
    fn from(value: PeakFitArgs<'a, 'b>) -> Self {
        (value.time, value.intensity).into()
    }
}

impl<'a, X, Y> From<&'a mzpeaks::feature::Feature<X, Y>> for PeakFitArgs<'a, 'a> {
    fn from(value: &'a mzpeaks::feature::Feature<X, Y>) -> Self {
        Self::from((value.time_view(), value.intensity_view()))
    }
}

impl<'a, X, Y> From<&'a mzpeaks::feature::FeatureView<'a, X, Y>> for PeakFitArgs<'a, 'a> {
    fn from(value: &'a mzpeaks::feature::FeatureView<X, Y>) -> Self {
        Self::from((value.time_view(), value.intensity_view()))
    }
}

impl<'a, X, Y> From<&'a mzpeaks::feature::SimpleFeature<X, Y>> for PeakFitArgs<'a, 'a> {
    fn from(value: &'a mzpeaks::feature::SimpleFeature<X, Y>) -> Self {
        Self::from((value.time_view(), value.intensity_view()))
    }
}

impl<'a, X, Y> From<&'a mzpeaks::feature::SimpleFeatureView<'a, X, Y>> for PeakFitArgs<'a, 'a> {
    fn from(value: &'a mzpeaks::feature::SimpleFeatureView<X, Y>) -> Self {
        Self::from((value.time_view(), value.intensity_view()))
    }
}

impl<'a, X, Y> From<&'a mzpeaks::feature::ChargedFeature<X, Y>> for PeakFitArgs<'a, 'a> {
    fn from(value: &'a mzpeaks::feature::ChargedFeature<X, Y>) -> Self {
        value.as_inner().0.into()
    }
}

impl<'a, X, Y> From<&'a mzpeaks::feature::ChargedFeatureView<'a, X, Y>> for PeakFitArgs<'a, 'a> {
    fn from(value: &'a mzpeaks::feature::ChargedFeatureView<X, Y>) -> Self {
        value.as_inner().0.into()
    }
}

/// Fit peak shapes on implementing types
pub trait FitPeaksOn<'a>
where
    PeakFitArgs<'a, 'a>: From<&'a Self>,
    Self: 'a,
{
    fn as_peak_shape_args(&'a self) -> PeakFitArgs<'a, 'a> {
        let data: PeakFitArgs<'a, 'a> = PeakFitArgs::from(self);
        data
    }

    /// Fit multiple peak models on this signal.
    fn fit_peaks_with(&'a self, config: FitConfig) -> SplittingPeakShapeFitter<'a, 'a> {
        let data: PeakFitArgs<'a, 'a> = self.as_peak_shape_args();
        let mut model = SplittingPeakShapeFitter::new(data);
        model.fit_with(config);
        model
    }
}

impl<'a, X: 'a, Y: 'a> FitPeaksOn<'a> for mzpeaks::feature::Feature<X, Y> {}
impl<'a, X: 'a, Y: 'a> FitPeaksOn<'a> for mzpeaks::feature::FeatureView<'a, X, Y> {}
impl<'a, X: 'a, Y: 'a> FitPeaksOn<'a> for mzpeaks::feature::SimpleFeature<X, Y> {}
impl<'a, X: 'a, Y: 'a> FitPeaksOn<'a> for mzpeaks::feature::SimpleFeatureView<'a, X, Y> {}
impl<'a, X: 'a, Y: 'a> FitPeaksOn<'a> for mzpeaks::feature::ChargedFeature<X, Y> {}
impl<'a, X: 'a, Y: 'a> FitPeaksOn<'a> for mzpeaks::feature::ChargedFeatureView<'a, X, Y> {}

/// Hyperparameters for fitting a peak shape model
#[derive(Debug, Clone)]
pub struct FitConfig {
    /// The maximum number of iterations to attempt when fitting a peak model
    max_iter: usize,
    /// The rate at which model parameters are updated
    learning_rate: f64,
    /// The minimum distance between the current loss and the previous loss at which to decide the model
    /// has converged
    convergence: f64,
    /// How much smoothing to perform before fitting a peak model.
    ///
    /// See [`PeakFitArgs::smooth`]
    smooth: usize,
}

impl FitConfig {
    /// The maximum number of iterations to attempt when fitting a peak model
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// The rate at which model parameters are updated
    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// The minimum distance between the current loss and the previous loss at which to decide the model
    /// has converged
    pub fn convergence(mut self, convergence: f64) -> Self {
        self.convergence = convergence;
        self
    }

    /// How much smoothing to perform before fitting a peak model.
    ///
    /// See [`PeakFitArgs::smooth`]
    pub fn smooth(mut self, smooth: usize) -> Self {
        self.smooth = smooth;
        self
    }
}

impl Default for FitConfig {
    fn default() -> Self {
        Self {
            max_iter: 50_000,
            learning_rate: 1e-3,
            convergence: 1e-6,
            smooth: 0,
        }
    }
}

/// Describe a model fitting procedure's output
#[derive(Debug, Default, Clone, Copy)]
pub struct ModelFitResult {
    /// The loss at the end of the optimization run
    pub loss: f64,
    /// The number of iterations run
    pub iterations: usize,
    /// Whether or not the model converged within the specified number of iterations
    pub converged: bool,
    /// Whether or not the model was able to fit *at all*
    pub success: bool,
}

impl ModelFitResult {
    pub fn new(loss: f64, iterations: usize, converged: bool, success: bool) -> Self {
        Self {
            loss,
            iterations,
            converged,
            success,
        }
    }
}

/// A set of peak shape model fitting behaviors that interacts with the [`PeakShapeModel`]
/// trait associated with some peak signal data.
pub trait PeakShapeModelFitter<'a, 'b> {
    /// The [`PeakShapeModel`] that this type will fit.
    type ModelType: PeakShapeModel + Debug;

    /// Construct a new [`PeakShapeModelFitter`] from [`PeakFitArgs`]
    fn from_args(args: PeakFitArgs<'a, 'b>) -> Self;

    /// Compute the model gradient against the enclosed data
    ///
    /// # See also
    /// [`PeakShapeModel::gradient`]
    fn gradient(&self, params: &Self::ModelType) -> Self::ModelType;

    /// Compute the model loss function the enclosed data
    ///
    /// # See also
    /// [`PeakShapeModel::loss`]
    fn loss(&self, params: &Self::ModelType) -> f64;

    /// Borrow the enclosed data
    fn data(&self) -> &PeakFitArgs;

    /// Iterate over the enclosed data
    fn iter(&self) -> PeakFitArgsIter {
        self.data().iter()
    }

    /// Compute the model score against the enclosed data
    ///
    /// # See also
    /// [`PeakShapeModel::score`]
    fn score(&self, model_params: &Self::ModelType) -> f64 {
        model_params.score(self.data())
    }

    /// Do the actual model fitting on the enclosed data.
    ///
    /// By default, this will use the model which produces
    /// the best loss.
    fn fit_model(
        &mut self,
        model_params: &mut Self::ModelType,
        config: FitConfig,
    ) -> ModelFitResult {
        let mut params = model_params.clone();

        let mut last_loss = f64::INFINITY;
        let mut best_loss = f64::INFINITY;
        let mut best_params = model_params.clone();
        let mut iters = 0;
        let mut converged = false;
        let mut success = true;

        let data = if config.smooth > 0 {
            self.data().smooth(config.smooth)
        } else {
            self.data().borrow()
        };

        for it in 0..config.max_iter {
            iters = it;
            let loss = params.loss(&data);
            let gradient = params.gradient(&data);

            log::trace!("{it}: Loss = {loss:0.3}: Gradient = {gradient:?}");
            params.gradient_update(gradient, config.learning_rate);
            if loss < best_loss {
                log::trace!("{it}: Updating best parameters {params:?}");
                best_loss = loss;
                best_params = params.clone();
            }

            if ((last_loss - loss).abs()) / loss < config.convergence {
                log::trace!("{it}: Convergence = {}", last_loss - loss);
                converged = true;
                break;
            }
            last_loss = loss;

            if loss.is_nan() || loss.is_infinite() {
                log::trace!("{it}: Aborting, loss invalid!");
                success = false;
                break;
            }
        }
        *model_params = best_params;
        ModelFitResult::new(best_loss, iters, converged, success)
    }
}

/// A model of an elution profile peak shape that can be estimated using gradient descent
pub trait PeakShapeModel: Clone {
    type Fitter<'a, 'b>: PeakShapeModelFitter<'a, 'b, ModelType = Self>;

    /// Compute the theoretical intensity at a specified coordinate
    ///
    /// # See also
    /// [`PeakShapeModel::predict`]
    /// [`PeakShapeModel::predict_iter`]
    fn density(&self, x: f64) -> f64;

    /// Update the parameters of the model based upon the `gradient` and a
    /// given learning rate.
    fn gradient_update(&mut self, gradient: Self, learning_rate: f64);

    /// Given a coordinate sequence, produce the complementary sequence of theoretical intensities
    ///
    /// # See also
    /// [`PeakShapeModel::density`]
    /// [`PeakShapeModel::predict_iter`]
    fn predict(&self, times: &[f64]) -> Vec<f64> {
        times.iter().map(|t| self.density(*t)).collect()
    }

    /// Given a coordinate iterator, produce the complementary iterator of theoretical intensities
    ///
    /// # See also
    /// [`PeakShapeModel::density`]
    /// [`PeakShapeModel::predict`]
    fn predict_iter<I: IntoIterator<Item = f64>>(&self, times: I) -> impl Iterator<Item = f64> {
        times.into_iter().map(|t| self.density(t))
    }

    /// Compute the gradient of the loss function for parameter optimization.
    fn gradient(&self, data: &PeakFitArgs) -> Self;

    /// Compute the loss function for optimization, mean-squared error
    fn loss(&self, data: &PeakFitArgs) -> f64 {
        data.iter()
            .map(|(t, i)| (i - self.density(t)).powi(2))
            .sum::<f64>()
            / data.len() as f64
    }

    /// Compute the difference between the observed signal and the theoretical signal,
    /// clamping the value to be non-negative
    fn residuals<'a, 'b, 'e: 'a + 'b>(&self, data: &'e PeakFitArgs<'a, 'b>) -> PeakFitArgs<'a, 'b> {
        let mut data = data.borrow();
        for (yhat, y) in self
            .predict_iter(data.time.iter().copied())
            .zip(data.intensity.to_mut().iter_mut())
        {
            *y -= yhat as f32;
            if *y < 0.0 {
                *y = 0.0;
            }
        }
        data
    }

    /// Compute the 1 - ratio of the peak shape model squared error to
    /// a straight line linear model squared error.
    ///
    /// This value is 0 when the ordinary linear model is much better than the peak
    /// shape model, and approaches 1.0 when the peak shape model is a much better fit
    /// of the data than straight line model.
    ///
    /// *NOTE*: The function output is clamped to the $`[0, 1]`$ range for consistency
    fn score(&self, data: &PeakFitArgs) -> f64 {
        let linear_resid = data.linear_residuals();
        let mut shape_resid = 0.0;

        for (x, y) in data.iter() {
            shape_resid += (y - self.density(x)).powi(2);
        }

        let line_test = shape_resid
            / (if linear_resid > 0.0 {
                linear_resid
            } else {
                1.0
            });
        (1.0 - line_test.max(1e-5)).max(0.0).min(1.0)
    }

    /// Given observed data, compute some initial parameters.
    ///
    /// This is the preferred means of producing an initial model
    /// for some data, prior to fitting.
    fn guess(data: &PeakFitArgs) -> Self;

    /// Fit the peak shape model to some data using the default
    /// [`FitConfig`] settings.
    fn fit(&mut self, data: PeakFitArgs) -> ModelFitResult {
        self.fit_with(data, Default::default())
    }

    /// Fit the peak shape model to some data using `config` options
    fn fit_with(&mut self, args: PeakFitArgs, config: FitConfig) -> ModelFitResult {
        let mut fitter = Self::Fitter::from_args(args);
        fitter.fit_model(self, config)
    }
}

/// Gaussian peak shape model
///
/// ```math
/// y = a\exp\left({\frac{-(\mu - x)^2}{2\sigma^2}}\right)
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct GaussianPeakShape {
    pub mu: f64,
    pub sigma: f64,
    pub amplitude: f64,
}

impl GaussianPeakShape {
    pub fn new(mu: f64, sigma: f64, amplitude: f64) -> Self {
        Self {
            mu,
            sigma,
            amplitude,
        }
    }

    /// Given observed data, compute some initial parameters
    pub fn guess(data: &PeakFitArgs) -> Self {
        if data.len() == 0 {
            return Self::new(1.0, 1.0, 1.0);
        }
        let idx = data.argmax();
        let mu = data.time[idx];
        let amplitude = data.intensity[idx] as f64;
        let sigma = 1.0;
        Self::new(mu, sigma, amplitude)
    }

    /// Compute the regularization term for the loss function
    pub fn regularization(&self) -> f64 {
        self.mu + self.sigma
    }

    /// Compute the loss function for optimization, mean-squared error
    pub fn loss(&self, data: &PeakFitArgs) -> f64 {
        data.iter()
            .map(|(t, i)| (i - self.density(t)).powi(2))
            .sum::<f64>()
            / data.len() as f64
            + self.regularization()
    }

    pub fn density(&self, x: f64) -> f64 {
        self.amplitude * (-0.5 * (x - self.mu).powi(2) / self.sigma.powi(2)).exp()
    }

    pub fn gradient(&self, data: &PeakFitArgs) -> Self {
        let amp = self.amplitude;
        let mu = self.mu;
        let sigma = self.sigma;

        let two_mu = 2.0 * mu;
        let sigma_squared = sigma.powi(2);
        let sigma_cubed = sigma.powi(3);
        let sigma_squared_inv = 1.0 / sigma_squared;

        let mut gradient_mu = 0.0;
        let mut gradient_sigma = 0.0;
        let mut gradient_amplitude = 0.0;

        for (x, y) in data.iter() {
            let mu_sub_x_squared = (-mu + x).powi(2);
            let half_mu_sub_x_squared_div_sigma_squared =
                -0.5 * mu_sub_x_squared * sigma_squared_inv;
            let half_mu_sub_x_squared_div_sigma_squared_exp =
                half_mu_sub_x_squared_div_sigma_squared.exp();

            let delta_y = -amp * half_mu_sub_x_squared_div_sigma_squared_exp + y;

            let delta_y_half_mu_sub_x_squared_div_sigma_squared_exp =
                delta_y * half_mu_sub_x_squared_div_sigma_squared_exp;

            gradient_mu += amp
                * (two_mu - 2.0 * x)
                * delta_y_half_mu_sub_x_squared_div_sigma_squared_exp
                * sigma_squared_inv
                + 1.0;

            gradient_sigma +=
                -2.0 * amp * mu_sub_x_squared * delta_y_half_mu_sub_x_squared_div_sigma_squared_exp
                    / sigma_cubed
                    + 1.0;

            gradient_amplitude += -2.0 * delta_y_half_mu_sub_x_squared_div_sigma_squared_exp;
        }

        let n = data.len() as f64;

        Self::new(gradient_mu / n, gradient_sigma / n, gradient_amplitude / n).gradient_norm()
    }

    fn gradient_norm(&self) -> Self {
        let mut g = [self.mu, self.sigma, self.amplitude];
        let gradnorm: f64 = g.iter().map(|f| f.abs()).sum::<f64>() / g.len() as f64;
        if gradnorm > 1.0 {
            g[0] /= gradnorm;
            g[1] /= gradnorm;
        }

        Self::new(g[0], g[1], g[2])
    }

    /// Compute the gradient w.r.t. $`\mu`$
    ///
    /// ```math
    /// -\frac{a \left(2 \mu - 2 x\right) \left(- a e^{- \frac{\left(- \mu + x\right)^{2}}{2 \sigma^{2}}} + y\right) e^{- \frac{\left(- \mu + x\right)^{2}}{2 \sigma^{2}}}}{\sigma^{2}} + 1
    /// ```
    fn mu_gradient(&self, data: &PeakFitArgs) -> f64 {
        let amp = self.amplitude;
        let mu = self.mu;
        let sigma = self.sigma;

        let two_mu = 2.0 * mu;
        let sigma_squared = sigma.powi(2);
        let sigma_squared_inv = 1.0 / sigma_squared;

        let grad: f64 = data
            .iter()
            .map(|(x, y)| {
                let mu_sub_x_squared = (-mu + x).powi(2);
                let half_mu_sub_x_squared_div_sigma_squared =
                    -0.5 * mu_sub_x_squared * sigma_squared_inv;
                let half_mu_sub_x_squared_div_sigma_squared_exp =
                    half_mu_sub_x_squared_div_sigma_squared.exp();

                amp * (two_mu - 2.0 * x)
                    * (-amp * half_mu_sub_x_squared_div_sigma_squared_exp + y)
                    * half_mu_sub_x_squared_div_sigma_squared_exp
                    * sigma_squared_inv
                    + 1.0
            })
            .sum();

        grad / data.len() as f64
    }

    /// Compute the gradient w.r.t. $`\sigma`$
    ///
    /// ```math
    /// - \frac{2 a \left(- \mu + x\right)^{2} \left(- a e^{- \frac{\left(- \mu + x\right)^{2}}{2 \sigma^{2}}} + y\right) e^{- \frac{\left(- \mu + x\right)^{2}}{2 \sigma^{2}}}}{\sigma^{3}} + 1
    /// ```
    fn sigma_gradient(&self, data: &PeakFitArgs) -> f64 {
        let amp = self.amplitude;
        let mu = self.mu;
        let sigma = self.sigma;

        let sigma_squared = sigma.powi(2);
        let sigma_cubed = sigma.powi(3);

        let grad: f64 = data
            .iter()
            .map(|(x, y)| {
                let mu_sub_x_squared = (-mu + x).powi(2);
                let half_mu_sub_x_squared_div_sigma_squared =
                    -0.5 * mu_sub_x_squared / sigma_squared;
                let half_mu_sub_x_squared_div_sigma_squared_exp =
                    half_mu_sub_x_squared_div_sigma_squared.exp();
                -2.0 * amp
                    * mu_sub_x_squared
                    * (-amp * half_mu_sub_x_squared_div_sigma_squared_exp + y)
                    * half_mu_sub_x_squared_div_sigma_squared_exp
                    / sigma_cubed
                    + 1.0
            })
            .sum();

        grad / data.len() as f64
    }

    /// Compute the gradient w.r.t. amplitude $`a`$
    ///
    /// ```math
    /// - 2 \left(- a e^{- \frac{\left(- \mu + x\right)^{2}}{2 \sigma^{2}}} + y\right) e^{- \frac{\left(- \mu + x\right)^{2}}{2 \sigma^{2}}}
    /// ```
    fn amplitude_gradient(&self, data: &PeakFitArgs) -> f64 {
        let amp = self.amplitude;
        let mu = self.mu;
        let sigma = self.sigma;

        let sigma_squared = sigma.powi(2);

        let grad: f64 = data
            .iter()
            .map(|(x, y)| {
                let mu_sub_x_squared = (-mu + x).powi(2);
                let half_mu_sub_x_squared_div_sigma_squared =
                    -0.5 * mu_sub_x_squared / sigma_squared;
                let half_mu_sub_x_squared_div_sigma_squared_exp =
                    half_mu_sub_x_squared_div_sigma_squared.exp();
                -2f64
                    * (-amp * half_mu_sub_x_squared_div_sigma_squared_exp + y)
                    * half_mu_sub_x_squared_div_sigma_squared_exp
            })
            .sum();

        grad / data.len() as f64
    }

    /// A non-optimized version of the gradient calculation used for testing
    /// correctness
    pub fn gradient_split(&self, data: &PeakFitArgs) -> Self {
        Self::new(
            self.mu_gradient(&data),
            self.sigma_gradient(&data),
            self.amplitude_gradient(&data),
        )
        .gradient_norm()
    }

    /// Update the parameters of the model based upon the `gradient` and a
    /// given learning rate.
    pub fn gradient_update(&mut self, gradient: Self, learning_rate: f64) {
        self.mu -= gradient.mu * learning_rate;
        self.sigma -= gradient.sigma * learning_rate;
        self.amplitude -= gradient.amplitude * learning_rate;
    }
}

impl PeakShapeModel for GaussianPeakShape {
    type Fitter<'a, 'b> = PeakShapeFitter<'a, 'b, Self>;

    fn density(&self, x: f64) -> f64 {
        self.density(x)
    }

    fn gradient_update(&mut self, gradient: Self, learning_rate: f64) {
        self.gradient_update(gradient, learning_rate);
    }

    fn guess(args: &PeakFitArgs) -> Self {
        Self::guess(args)
    }

    fn gradient(&self, data: &PeakFitArgs) -> Self {
        self.gradient(data)
    }

    fn loss(&self, data: &PeakFitArgs) -> f64 {
        data.iter()
            .map(|(t, i)| (i - self.density(t)).powi(2))
            .sum::<f64>()
            / data.len() as f64
            + self.regularization()
    }
}

/// Skewed Gaussian peak shape model
///
/// ```math
/// y = a\left(\text{erf}\left({\sqrt{2} \lambda\frac{\mu - x}{2\sigma}}\right) + 1\right)\exp\left(-\frac{(\mu-x)^2}{2\sigma^2}\right)
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct SkewedGaussianPeakShape {
    pub mu: f64,
    pub sigma: f64,
    pub amplitude: f64,
    pub lambda: f64,
}

impl SkewedGaussianPeakShape {
    pub fn new(mu: f64, sigma: f64, amplitude: f64, lambda: f64) -> Self {
        Self {
            mu,
            sigma,
            amplitude,
            lambda,
        }
    }

    /// Given observed data, compute some initial parameters
    pub fn guess(data: &PeakFitArgs) -> Self {
        if data.len() == 0 {
            return Self::new(1.0, 1.0, 1.0, 1.0);
        }
        let idx = data.argmax();
        let mu = data.time[idx];
        let amplitude = data.intensity[idx] as f64;
        let sigma = 1.0;
        let lambda = 1.0;
        Self::new(mu, sigma, amplitude, lambda)
    }

    /// Compute the theoretical intensity at a specified coordinate
    pub fn density(&self, x: f64) -> f64 {
        self.amplitude
            * (erf(SQRT_2 * self.lambda * (-self.mu + x) / (2.0 * self.sigma)) + 1.0)
            * (-0.5 * (-self.mu + x).powi(2) / self.sigma.powi(2)).exp()
    }

    /// Compute the regularization term for the loss function
    pub fn regularization(&self) -> f64 {
        self.mu + self.sigma + self.lambda
    }

    /// A non-optimized version of the gradient calculation used for testing
    /// correctness
    pub fn gradient_split(&self, data: &PeakFitArgs) -> Self {
        Self::new(
            self.mu_gradient(&data),
            self.sigma_gradient(&data),
            self.amplitude_gradient(&data),
            self.lambda_gradient(&data),
        )
        .gradient_norm()
    }

    fn gradient_norm(&self) -> Self {
        let mut g = [self.mu, self.sigma, self.amplitude, self.lambda];
        let gradnorm: f64 = g.iter().map(|f| f.abs()).sum::<f64>() / g.len() as f64;
        if gradnorm > 1.0 {
            g[0] /= gradnorm;
            g[1] /= gradnorm;
            g[3] /= gradnorm;
        }

        SkewedGaussianPeakShape::new(g[0], g[1], g[2], g[3])
    }

    /// Compute the gradient of the loss function for parameter optimization.
    pub fn gradient(&self, data: &PeakFitArgs) -> Self {
        let amp = self.amplitude;
        let mu = self.mu;
        let sigma = self.sigma;
        let lam = self.lambda;

        let two_sigma = sigma * 2.0;
        let sigma_square = sigma.powi(2);
        let sigma_cubed = sigma.powi(3);
        let skew = 2.0 * 1.4142135623731 * amp * lam;
        let delta_skew = -2.0 * 1.4142135623731 * amp;
        let sqrt_2_lam = SQRT_2 * lam;
        let sqrt_pi_sigma = PI.sqrt() * sigma;
        let sqrt_pi_sigma_square = PI.sqrt() * sigma_square;
        let neg_half_lam_squared = -1_f64 / 2.0 * lam.powi(2);

        let mut gradient_mu = 0.0;
        let mut gradient_sigma = 0.0;
        let mut gradient_lambda = 0.0;
        let mut gradient_amplitude = 0.0;

        for (x, y) in data.iter() {
            let mu_sub_x = -mu + x;
            let mu_sub_x_squared = mu_sub_x.powi(2);
            let neg_half_mu_sub_x_squared_div_sigma_squared_exp =
                (-0.5 * mu_sub_x_squared / sigma_square).exp();
            let erf_sqrt_2_lam_mu_sub_x_div_two_sigma_plus_one =
                erf(sqrt_2_lam * mu_sub_x / two_sigma) + 1.0;
            let neg_half_lam_squared_mu_sub_x_squared_div_sigma_square_exp =
                (neg_half_lam_squared * mu_sub_x_squared / sigma_square).exp();

            let delta_y = -amp
                * erf_sqrt_2_lam_mu_sub_x_div_two_sigma_plus_one
                * neg_half_mu_sub_x_squared_div_sigma_squared_exp
                + y;

            gradient_mu += delta_y
                * (skew
                    * neg_half_mu_sub_x_squared_div_sigma_squared_exp
                    * neg_half_lam_squared_mu_sub_x_squared_div_sigma_square_exp
                    / sqrt_pi_sigma
                    + amp
                        * (2.0 * mu - 2.0 * x)
                        * erf_sqrt_2_lam_mu_sub_x_div_two_sigma_plus_one
                        * neg_half_mu_sub_x_squared_div_sigma_squared_exp
                        / sigma_square)
                + 1.0;

            gradient_sigma += delta_y
                * (skew
                    * mu_sub_x
                    * neg_half_mu_sub_x_squared_div_sigma_squared_exp
                    * neg_half_lam_squared_mu_sub_x_squared_div_sigma_square_exp
                    / (sqrt_pi_sigma_square)
                    - 2.0
                        * amp
                        * mu_sub_x_squared
                        * erf_sqrt_2_lam_mu_sub_x_div_two_sigma_plus_one
                        * neg_half_mu_sub_x_squared_div_sigma_squared_exp
                        / sigma_cubed)
                + 1.0;

            gradient_lambda += delta_skew
                * mu_sub_x
                * delta_y
                * neg_half_mu_sub_x_squared_div_sigma_squared_exp
                * neg_half_lam_squared_mu_sub_x_squared_div_sigma_square_exp
                / sqrt_pi_sigma
                + 1.0;

            gradient_amplitude += -2.0
                * delta_y
                * erf_sqrt_2_lam_mu_sub_x_div_two_sigma_plus_one
                * neg_half_mu_sub_x_squared_div_sigma_squared_exp
        }

        let n = data.len() as f64;

        Self::new(
            gradient_mu / n,
            gradient_sigma / n,
            gradient_amplitude / n,
            gradient_lambda / n,
        )
        .gradient_norm()
    }

    fn mu_gradient(&self, data: &PeakFitArgs) -> f64 {
        let amp = self.amplitude;
        let mu = self.mu;
        let sigma = self.sigma;
        let lam = self.lambda;

        let two_sigma = sigma * 2.0;
        let sigma_square = sigma.powi(2);
        let skew = 2.0 * 1.4142135623731 * amp * lam;
        let sqrt_2_lam = SQRT_2 * lam;
        let sqrt_pi_sigma = PI.sqrt() * sigma;
        let neg_half_lam_squared = -1_f64 / 2.0 * lam.powi(2);

        let mut grad = 0.0;
        for (x, y) in data.iter() {
            grad += (-amp
                * (erf(sqrt_2_lam * (-mu + x) / (two_sigma)) + 1.0)
                * (-0.5 * (-mu + x).powi(2) / sigma_square).exp()
                + y)
                * (skew
                    * (-0.5 * (-mu + x).powi(2) / sigma_square).exp()
                    * (neg_half_lam_squared * (-mu + x).powi(2) / sigma_square).exp()
                    / sqrt_pi_sigma
                    + amp
                        * (2.0 * mu - 2.0 * x)
                        * (erf(sqrt_2_lam * (-mu + x) / (two_sigma)) + 1.0)
                        * (-0.5 * (-mu + x).powi(2) / sigma_square).exp()
                        / sigma_square)
                + 1.0
        }
        grad / data.len() as f64
    }

    fn sigma_gradient(&self, data: &PeakFitArgs) -> f64 {
        let amp = self.amplitude;
        let mu = self.mu;
        let sigma = self.sigma;
        let lam = self.lambda;

        let two_sigma = sigma * 2.0;
        let sigma_square = sigma.powi(2);
        let sigma_cubed = sigma.powi(3);
        let skew = 2.0 * 1.4142135623731 * amp * lam;
        let sqrt_2_lam = SQRT_2 * lam;
        let sqrt_pi_sigma_square = PI.sqrt() * sigma_square;
        let neg_half_lam_squared = -1_f64 / 2.0 * lam.powi(2);

        let mut grad = 0.0;
        for (x, y) in data.iter() {
            grad += (-amp
                * (erf(sqrt_2_lam * (-mu + x) / two_sigma) + 1.0)
                * (-0.5 * (-mu + x).powi(2) / sigma_square).exp()
                + y)
                * (skew
                    * (-mu + x)
                    * (-0.5 * (-mu + x).powi(2) / sigma_square).exp()
                    * (neg_half_lam_squared * (-mu + x).powi(2) / sigma_square).exp()
                    / (sqrt_pi_sigma_square)
                    - 2.0
                        * amp
                        * (-mu + x).powi(2)
                        * (erf(sqrt_2_lam * (-mu + x) / two_sigma) + 1.0)
                        * (-0.5 * (-mu + x).powi(2) / sigma_square).exp()
                        / sigma_cubed)
                + 1.0
        }
        grad / data.len() as f64
    }

    fn amplitude_gradient(&self, data: &PeakFitArgs) -> f64 {
        let amp = self.amplitude;
        let mu = self.mu;
        let sigma = self.sigma;
        let lam = self.lambda;

        let two_sigma = sigma * 2.0;
        let sigma_square = sigma.powi(2);
        let sqrt_2_lam = SQRT_2 * lam;

        let mut grad: f64 = 0.0;
        for (x, y) in data.iter() {
            grad += -2.0
                * (-amp
                    * (erf(sqrt_2_lam * (-mu + x) / (two_sigma)) + 1.0)
                    * (-0.5 * (-mu + x).powi(2) / sigma_square).exp()
                    + y)
                * (erf(sqrt_2_lam * (-mu + x) / (two_sigma)) + 1.0)
                * (-0.5 * (-mu + x).powi(2) / sigma_square).exp()
        }
        grad / data.len() as f64
    }

    fn lambda_gradient(&self, data: &PeakFitArgs) -> f64 {
        let amp = self.amplitude;
        let mu = self.mu;
        let sigma = self.sigma;
        let lam = self.lambda;

        let two_sigma = sigma * 2.0;
        let sigma_square = sigma.powi(2);
        let delta_skew = -2.0 * 1.4142135623731 * amp;
        let sqrt_2_lam = SQRT_2 * lam;
        let sqrt_pi_sigma = PI.sqrt() * sigma;
        let neg_half_lam_squared = -1_f64 / 2.0 * lam.powi(2);

        let mut grad = 0.0;
        for (x, y) in data.iter() {
            let mu_sub_x = -mu + x;
            let mu_sub_x_squared = mu_sub_x.powi(2);
            let neg_half_mu_sub_x_squared_div_sigma_squared =
                (-0.5 * mu_sub_x_squared / sigma_square).exp();

            grad += delta_skew
                * mu_sub_x
                * (-amp
                    * (erf(sqrt_2_lam * mu_sub_x / (two_sigma)) + 1.0)
                    * neg_half_mu_sub_x_squared_div_sigma_squared
                    + y)
                * neg_half_mu_sub_x_squared_div_sigma_squared
                * (neg_half_lam_squared * mu_sub_x_squared / sigma_square).exp()
                / sqrt_pi_sigma
                + 1.0
        }
        grad / data.len() as f64
    }

    /// Update the parameters of the model based upon the `gradient` and a
    /// given learning rate.
    pub fn gradient_update(&mut self, gradient: Self, learning_rate: f64) {
        self.mu -= gradient.mu * learning_rate;
        self.sigma -= gradient.sigma * learning_rate;
        self.amplitude -= gradient.amplitude * learning_rate;
        if self.amplitude < 0.0 {
            self.amplitude = 0.0
        }
        self.lambda -= gradient.lambda * learning_rate;
    }
}

impl PeakShapeModel for SkewedGaussianPeakShape {
    type Fitter<'a, 'b> = PeakShapeFitter<'a, 'b, Self>;

    fn density(&self, x: f64) -> f64 {
        self.density(x)
    }

    fn gradient_update(&mut self, gradient: Self, learning_rate: f64) {
        self.gradient_update(gradient, learning_rate);
    }

    fn guess(args: &PeakFitArgs) -> Self {
        Self::guess(args)
    }

    fn gradient(&self, data: &PeakFitArgs) -> Self {
        self.gradient(data)
    }

    /// Compute the loss function for optimization, mean-squared error
    fn loss(&self, data: &PeakFitArgs) -> f64 {
        data.iter()
            .map(|(t, i)| (i - self.density(t)).powi(2))
            .sum::<f64>()
            / data.len() as f64
            + self.regularization()
    }
}

/// Bi-Gaussian peak shape model
///
/// ```math
/// y = \begin{cases}
///     a\exp\left({\frac{-(\mu - x)^2}{2\sigma_a^2}}\right) & x \le \mu \\
///     a\exp\left({\frac{-(\mu - x)^2}{2\sigma_b^2}}\right) & x \gt x
/// \end{cases}
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct BiGaussianPeakShape {
    pub mu: f64,
    pub sigma_falling: f64,
    pub sigma_rising: f64,
    pub amplitude: f64,
}

impl BiGaussianPeakShape {
    pub fn new(mu: f64, sigma_low: f64, sigma_high: f64, amplitude: f64) -> Self {
        Self {
            mu,
            sigma_falling: sigma_low,
            sigma_rising: sigma_high,
            amplitude,
        }
    }

    /// Given observed data, compute some initial parameters
    pub fn guess(data: &PeakFitArgs) -> Self {
        if data.len() == 0 {
            return Self::new(1.0, 1.0, 1.0, 1.0);
        }
        let idx = data.argmax();
        let mu = data.time[idx];
        let amplitude = data.intensity[idx] as f64;
        let sigma = 1.0;
        Self::new(mu, sigma, sigma, amplitude)
    }

    /// Compute the theoretical intensity at a specified coordinate
    pub fn density(&self, x: f64) -> f64 {
        if self.mu >= x {
            self.amplitude * (-0.5 * (-self.mu + x).powi(2) / self.sigma_falling.powi(2)).exp()
        } else {
            self.amplitude * (-0.5 * (-self.mu + x).powi(2) / self.sigma_rising.powi(2)).exp()
        }
    }

    /// Update the parameters of the model based upon the `gradient` and a
    /// given learning rate.
    pub fn gradient_update(&mut self, gradient: Self, learning_rate: f64) {
        self.mu -= gradient.mu * learning_rate;
        self.sigma_falling -= gradient.sigma_falling * learning_rate;
        self.sigma_rising -= gradient.sigma_rising * learning_rate;

        self.amplitude -= gradient.amplitude * learning_rate;
        if self.amplitude < 0.0 {
            self.amplitude = 0.0
        }
    }

    /// Compute the regularization term for the loss function
    pub fn regularization(&self) -> f64 {
        self.mu + self.sigma_falling + self.sigma_rising
    }

    /// Compute the gradient of the loss function for parameter optimization.
    pub fn gradient(&self, data: &PeakFitArgs) -> BiGaussianPeakShape {
        let mu = self.mu;
        let amp = self.amplitude;
        let sigma_low = self.sigma_falling;
        let sigma_high = self.sigma_rising;

        let sigma_low_squared = sigma_low.powi(2);
        let sigma_high_squared = sigma_high.powi(2);
        let sigma_low_cubed = sigma_low.powi(3);
        let sigma_high_cubed = sigma_high.powi(3);
        let two_mu = mu * 2.0;
        let neg_half_amp = -0.5 * amp;

        let mut gradient_mu = 0.0;
        let mut gradient_sigma_high = 0.0;
        let mut gradient_sigma_low = 0.0;
        let mut gradient_amplitude = 0.0;

        for (x, y) in data.iter() {
            let mu_sub_x_squared = (mu - x).powi(2);
            let neg_half_mu_sub_x_squared = -0.5 * mu_sub_x_squared;

            if mu >= x {
                let neg_half_mu_sub_x_squared_div_sigma_low_squared =
                    neg_half_mu_sub_x_squared / sigma_low_squared;

                let neg_half_mu_sub_x_squared_div_sigma_low_squared_exp =
                    neg_half_mu_sub_x_squared_div_sigma_low_squared.exp();
                let delta_y =
                    -2.0 * (y - amp * neg_half_mu_sub_x_squared_div_sigma_low_squared_exp);

                gradient_mu += delta_y
                    * (neg_half_amp
                        * (two_mu - 2.0 * x)
                        * neg_half_mu_sub_x_squared_div_sigma_low_squared_exp
                        / sigma_low_squared)
                    + 1.0;

                gradient_sigma_high += 1.0;

                gradient_sigma_low += delta_y
                    * (amp
                        * mu_sub_x_squared
                        * neg_half_mu_sub_x_squared_div_sigma_low_squared_exp
                        / sigma_low_cubed)
                    + 1.0;

                gradient_amplitude += delta_y * neg_half_mu_sub_x_squared_div_sigma_low_squared_exp
            } else {
                let neg_half_mu_sub_x_squared_div_sigma_high_squared =
                    neg_half_mu_sub_x_squared / sigma_high_squared;

                let neg_half_mu_sub_x_squared_div_sigma_high_squared_exp =
                    neg_half_mu_sub_x_squared_div_sigma_high_squared.exp();

                let delta_y =
                    -2.0 * (y - amp * neg_half_mu_sub_x_squared_div_sigma_high_squared_exp);

                gradient_mu += delta_y
                    * (neg_half_amp
                        * (two_mu - 2.0 * x)
                        * neg_half_mu_sub_x_squared_div_sigma_high_squared_exp
                        / sigma_high_squared)
                    + 1.0;

                gradient_sigma_high += delta_y
                    * (amp
                        * mu_sub_x_squared
                        * neg_half_mu_sub_x_squared_div_sigma_high_squared_exp
                        / sigma_high_cubed)
                    + 1.0;

                gradient_sigma_low += 1.0;

                gradient_amplitude += delta_y * neg_half_mu_sub_x_squared_div_sigma_high_squared_exp
            }
        }

        let n = data.len() as f64;

        BiGaussianPeakShape::new(
            gradient_mu / n,
            gradient_sigma_low / n,
            gradient_sigma_high / n,
            gradient_amplitude / n,
        )
        .gradient_norm()
    }

    /// A non-optimized version of the gradient calculation used for testing
    /// correctness
    pub fn gradient_split(&self, data: &PeakFitArgs) -> BiGaussianPeakShape {
        let g = Self::new(
            self.gradient_mu(&data),
            self.gradient_sigma_falling(&data),
            self.gradient_sigma_rising(&data),
            self.gradient_amplitude(&data),
        );
        g.gradient_norm()
    }

    fn gradient_norm(&self) -> Self {
        let mut g = [self.mu, self.sigma_falling, self.sigma_rising, self.amplitude];
        let gradnorm: f64 = g.iter().map(|f| f.abs()).sum::<f64>() / g.len() as f64;
        if gradnorm > 1.0 {
            g[0] /= gradnorm;
            g[1] /= gradnorm;
            g[2] /= gradnorm;
        }
        BiGaussianPeakShape::new(g[0], g[1], g[2], g[3])
    }

    fn gradient_mu(&self, data: &PeakFitArgs) -> f64 {
        let mu = self.mu;
        let amp = self.amplitude;
        let sigma_low = self.sigma_falling;
        let sigma_high = self.sigma_rising;

        let sigma_low_squared = sigma_low.powi(2);
        let sigma_high_squared = sigma_high.powi(2);
        let neg_half_amp = -0.5 * amp;

        data.iter()
            .map(|(x, y)| {
                let mu_sub_x_squared = (mu - x).powi(2);
                let neg_half_mu_sub_x_squared = -0.5 * mu_sub_x_squared;

                if mu >= x {
                    -2.0 * (y - amp * (neg_half_mu_sub_x_squared / sigma_low_squared).exp())
                        * (neg_half_amp
                            * (2.0 * mu - 2.0 * x)
                            * (neg_half_mu_sub_x_squared / sigma_low_squared).exp()
                            / sigma_low_squared)
                        + 1.0
                } else {
                    -2.0 * (y - amp * (neg_half_mu_sub_x_squared / sigma_high_squared).exp())
                        * (neg_half_amp
                            * (2.0 * mu - 2.0 * x)
                            * (neg_half_mu_sub_x_squared / sigma_high_squared).exp()
                            / sigma_high_squared)
                        + 1.0
                }
            })
            .sum::<f64>()
            / data.len() as f64
    }

    fn gradient_sigma_rising(&self, data: &PeakFitArgs) -> f64 {
        let mu = self.mu;
        let amp = self.amplitude;
        let sigma_low = self.sigma_falling;
        let sigma_high = self.sigma_rising;

        let sigma_low_squared = sigma_low.powi(2);
        let sigma_high_squared = sigma_high.powi(2);
        let sigma_high_cubed = sigma_high.powi(3);

        data.iter()
            .map(|(x, y)| {
                -2.0 * (y - if mu >= x {
                    amp * (-0.5 * (-mu + x).powi(2) / sigma_low_squared).exp()
                } else {
                    amp * (-0.5 * (-mu + x).powi(2) / sigma_high_squared).exp()
                }) * if mu >= x {
                    0.0
                } else {
                    amp * (-mu + x).powi(2) * (-0.5 * (-mu + x).powi(2) / sigma_high_squared).exp()
                        / sigma_high_cubed
                } + 1.0
            })
            .sum::<f64>()
            / data.len() as f64
    }

    fn gradient_sigma_falling(&self, data: &PeakFitArgs) -> f64 {
        let mu = self.mu;
        let amp = self.amplitude;
        let sigma_low = self.sigma_falling;
        let sigma_high = self.sigma_rising;

        let sigma_low_squared = sigma_low.powi(2);
        let sigma_high_squared = sigma_high.powi(2);
        let sigma_low_cubed = sigma_low.powi(3);

        data.iter()
            .map(|(x, y)| {
                -2.0 * (y - if mu >= x {
                    amp * (-0.5 * (-mu + x).powi(2) / sigma_low_squared).exp()
                } else {
                    amp * (-0.5 * (-mu + x).powi(2) / sigma_high_squared).exp()
                }) * if mu >= x {
                    amp * (-mu + x).powi(2) * (-0.5 * (-mu + x).powi(2) / sigma_low_squared).exp()
                        / sigma_low_cubed
                } else {
                    0.0
                } + 1.0
            })
            .sum::<f64>()
            / data.len() as f64
    }

    fn gradient_amplitude(&self, data: &PeakFitArgs) -> f64 {
        let mu = self.mu;
        let amp = self.amplitude;
        let sigma_low = self.sigma_falling;
        let sigma_high = self.sigma_rising;

        let sigma_low_squared = sigma_low.powi(2);
        let sigma_high_squared = sigma_high.powi(2);

        data.iter()
            .map(|(x, y)| {
                -2.0 * (y - if mu >= x {
                    amp * (-0.5 * (-mu + x).powi(2) / sigma_low_squared).exp()
                } else {
                    amp * (-0.5 * (-mu + x).powi(2) / sigma_high_squared).exp()
                }) * if mu >= x {
                    (-0.5 * (-mu + x).powi(2) / sigma_low_squared).exp()
                } else {
                    (-0.5 * (-mu + x).powi(2) / sigma_high_squared).exp()
                }
            })
            .sum::<f64>()
            / data.len() as f64
    }
}

impl PeakShapeModel for BiGaussianPeakShape {
    type Fitter<'a, 'b> = PeakShapeFitter<'a, 'b, Self>;

    fn density(&self, x: f64) -> f64 {
        self.density(x)
    }

    fn gradient_update(&mut self, gradient: Self, learning_rate: f64) {
        self.gradient_update(gradient, learning_rate);
    }

    fn guess(args: &PeakFitArgs) -> Self {
        Self::guess(args)
    }

    fn loss(&self, data: &PeakFitArgs) -> f64 {
        data.iter()
            .map(|(t, i)| (i - self.density(t)).powi(2))
            .sum::<f64>()
            / data.len() as f64
            + self.regularization()
    }

    fn gradient(&self, data: &PeakFitArgs) -> Self {
        self.gradient(data)
    }
}

/// Fit a single [`PeakShapeModel`] type
#[derive(Debug, Clone)]
pub struct PeakShapeFitter<'a, 'b, T: PeakShapeModel + Debug> {
    pub data: PeakFitArgs<'a, 'b>,
    pub model: Option<T>,
}

impl<'a, 'b, T: PeakShapeModel + Debug> PeakShapeModelFitter<'a, 'b>
    for PeakShapeFitter<'a, 'b, T>
{
    type ModelType = T;

    fn from_args(args: PeakFitArgs<'a, 'b>) -> Self {
        Self::new(args)
    }

    fn gradient(&self, params: &Self::ModelType) -> Self::ModelType {
        params.gradient(&self.data)
    }

    fn loss(&self, params: &Self::ModelType) -> f64 {
        params.loss(&self.data)
    }

    fn data(&self) -> &PeakFitArgs {
        &self.data
    }

    fn fit_model(
        &mut self,
        model_params: &mut Self::ModelType,
        config: FitConfig,
    ) -> ModelFitResult {
        let mut params = model_params.clone();

        let mut last_loss = f64::INFINITY;
        let mut best_loss = f64::INFINITY;
        let mut best_params = model_params.clone();
        let mut iters = 0;
        let mut converged = false;
        let mut success = true;

        let data = if config.smooth > 0 {
            self.data().smooth(config.smooth)
        } else {
            self.data().borrow()
        };

        for it in 0..config.max_iter {
            iters = it;
            let loss = params.loss(&data);
            let gradient = params.gradient(&data);

            log::trace!("{it}: Loss = {loss:0.3}: Gradient = {gradient:?}");
            params.gradient_update(gradient, config.learning_rate);
            if loss < best_loss {
                log::trace!("{it}: Updating best parameters {params:?}");
                best_loss = loss;
                best_params = params.clone();
            }

            if (last_loss - loss).abs() < config.convergence {
                log::trace!("{it}: Convergence = {}", last_loss - loss);
                converged = true;
                break;
            }
            last_loss = loss;

            if loss.is_nan() || loss.is_infinite() {
                log::trace!("{it}: Aborting, loss invalid!");
                success = false;
                break;
            }
        }

        self.model = Some(best_params.clone());
        *model_params = best_params;
        ModelFitResult::new(best_loss, iters, converged, success)
    }
}

impl<'a, 'b, T: PeakShapeModel + Debug> PeakShapeFitter<'a, 'b, T> {
    pub fn new(data: PeakFitArgs<'a, 'b>) -> Self {
        Self { data, model: None }
    }

    /// Compute the model residuals over the observed time axis
    pub fn residuals(&self) -> PeakFitArgs<'_, '_> {
        self.model.as_ref().unwrap().residuals(&self.data)
    }

    /// Create a synthetic signal profile using the observed time axis but use the model predicted signal
    /// magnitude.
    pub fn predicted(&self) -> PeakFitArgs<'_, '_> {
        let predicted = self
            .model
            .as_ref()
            .unwrap()
            .predict(&self.data.time)
            .into_iter()
            .map(|x| x as f32)
            .collect();
        let mut dup = self.data.borrow();
        dup.intensity = Cow::Owned(predicted);
        dup
    }

    /// Compute the fitted model's score on the observed data
    ///
    /// # See also
    /// [`PeakShapeModel::score`]
    pub fn score(&self) -> f64 {
        self.model.as_ref().unwrap().score(&self.data)
    }
}

/// A dispatching peak shape model that can represent a variety of different
/// peak shapes.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub enum PeakShape {
    Gaussian(GaussianPeakShape),
    SkewedGaussian(SkewedGaussianPeakShape),
    BiGaussian(BiGaussianPeakShape),
}

macro_rules! dispatch_peak {
    ($d:ident, $r:ident, $e:expr) => {
        match $d {
            PeakShape::Gaussian($r) => $e,
            PeakShape::SkewedGaussian($r) => $e,
            PeakShape::BiGaussian($r) => $e,
        }
    };
}

impl From<GaussianPeakShape> for PeakShape {
    fn from(value: GaussianPeakShape) -> Self {
        Self::Gaussian(value)
    }
}

impl From<SkewedGaussianPeakShape> for PeakShape {
    fn from(value: SkewedGaussianPeakShape) -> Self {
        Self::SkewedGaussian(value)
    }
}

impl From<BiGaussianPeakShape> for PeakShape {
    fn from(value: BiGaussianPeakShape) -> Self {
        Self::BiGaussian(value)
    }
}

impl PeakShape {
    /// Guess an initial [`GaussianPeakShape`]
    pub fn gaussian(data: &PeakFitArgs) -> Self {
        Self::Gaussian(GaussianPeakShape::guess(data))
    }

    /// Guess an initial [`SkewedGaussianPeakShape`]
    pub fn skewed_gaussian(data: &PeakFitArgs) -> Self {
        Self::SkewedGaussian(SkewedGaussianPeakShape::guess(data))
    }

    /// Guess an initial [`BiGaussianPeakShape`]
    pub fn bigaussian(data: &PeakFitArgs) -> Self {
        Self::BiGaussian(BiGaussianPeakShape::guess(data))
    }

    /// Compute the theoretical intensity at a specified coordinate
    pub fn density(&self, x: f64) -> f64 {
        dispatch_peak!(self, p, p.density(x))
    }

    /// Given a coordinate sequence, produce the complementary sequence of theoretical intensities
    ///
    /// # See also
    /// [`PeakShape::density`]
    pub fn predict(&self, times: &[f64]) -> Vec<f64> {
        dispatch_peak!(self, p, p.predict(times))
    }

    /// Compute the difference between the observed signal and the theoretical signal,
    /// clamping the value to be non-negative
    ///
    /// # See also
    /// [`PeakShapeModel::residuals`]
    pub fn residuals<'a, 'b, 'e: 'a + 'b>(
        &self,
        data: &'e PeakFitArgs<'a, 'b>,
    ) -> PeakFitArgs<'a, 'b> {
        dispatch_peak!(self, p, p.residuals(data))
    }

    /// Compute the 1 - ratio of the peak shape model squared error to
    /// a straight line linear model squared error.
    ///
    /// # See also
    /// [`PeakShapeModel::score`]
    pub fn score(&self, data: &PeakFitArgs) -> f64 {
        dispatch_peak!(self, p, p.score(data))
    }

    /// Fit the peak shape model to some data using the default
    /// [`FitConfig`] settings.
    ///
    /// # See also
    /// [`PeakShapeModel::fit`]
    pub fn fit(&mut self, args: PeakFitArgs) -> ModelFitResult {
        dispatch_peak!(self, p, p.fit_with(args, Default::default()))
    }

    /// Fit the peak shape model to some data using `config` options
    ///
    /// # See also
    /// [`PeakShapeModel::fit_with`]
    pub fn fit_with(&mut self, args: PeakFitArgs, config: FitConfig) -> ModelFitResult {
        dispatch_peak!(self, p, p.fit_with(args, config))
    }
}

impl PeakShapeModel for PeakShape {
    type Fitter<'a, 'b> = PeakShapeFitter<'a, 'b, Self>;

    fn density(&self, x: f64) -> f64 {
        self.density(x)
    }

    fn gradient_update(&mut self, gradient: Self, learning_rate: f64) {
        match (self, gradient) {
            (Self::Gaussian(this), Self::Gaussian(gradient)) => {
                this.gradient_update(gradient, learning_rate);
            }
            (Self::BiGaussian(this), Self::BiGaussian(gradient)) => {
                this.gradient_update(gradient, learning_rate);
            }
            (Self::SkewedGaussian(this), Self::SkewedGaussian(gradient)) => {
                this.gradient_update(gradient, learning_rate);
            }
            (this, gradient) => panic!("Invalid gradient type {gradient:?} for model {this:?}"),
        };
    }

    fn gradient(&self, data: &PeakFitArgs) -> Self {
        match self {
            PeakShape::Gaussian(model) => Self::Gaussian(model.gradient(data)),
            PeakShape::SkewedGaussian(model) => Self::SkewedGaussian(model.gradient(data)),
            PeakShape::BiGaussian(model) => Self::BiGaussian(model.gradient(data)),
        }
    }

    fn guess(args: &PeakFitArgs) -> Self {
        Self::BiGaussian(BiGaussianPeakShape::guess(args))
    }
}

/// Represent a combination of multiple [`PeakShape`] models
#[derive(Debug, Default, Clone)]
pub struct MultiPeakShapeFit {
    fits: Vec<PeakShape>,
}

impl MultiPeakShapeFit {
    pub fn new(fits: Vec<PeakShape>) -> Self {
        Self { fits }
    }

    pub fn density(&self, x: f64) -> f64 {
        self.iter().map(|p| p.density(x)).sum()
    }

    pub fn predict(&self, times: &[f64]) -> Vec<f64> {
        times.iter().map(|t| self.density(*t)).collect()
    }

    pub fn residuals<'a, 'b, 'e: 'a + 'b>(
        &self,
        data: &'e PeakFitArgs<'a, 'b>,
    ) -> PeakFitArgs<'a, 'b> {
        let mut data = data.borrow();
        for (yhat, y) in data
            .time
            .iter()
            .copied()
            .map(|t| self.density(t))
            .zip(data.intensity.to_mut().iter_mut())
        {
            *y -= yhat as f32;
            // if *y < 0.0 {
            //     *y = 0.0;
            // }
        }
        data
    }

    pub fn iter(&self) -> std::slice::Iter<'_, PeakShape> {
        self.fits.iter()
    }

    pub fn score(&self, data: &PeakFitArgs<'_, '_>) -> f64 {
        let linear_resid = data.linear_residuals();
        let mut shape_resid = 0.0;

        for (x, y) in data.iter() {
            shape_resid += (y - self.density(x)).powi(2);
        }

        let line_test = shape_resid / linear_resid;
        1.0 - line_test.max(1e-5)
    }

    pub fn push(&mut self, model: PeakShape) {
        self.fits.push(model);
    }
}

/// Fitter for multiple peak shapes on the signal split across
/// multiple disjoint intervals.
///
/// This is preferred for "real world data" which may not be
/// well behaved signal.
#[derive(Debug, Clone)]
pub struct SplittingPeakShapeFitter<'a, 'b> {
    pub data: PeakFitArgs<'a, 'b>,
    pub peak_fits: MultiPeakShapeFit,
}

impl<'a> SplittingPeakShapeFitter<'a, 'a> {
    pub fn new(data: PeakFitArgs<'a, 'a>) -> Self {
        Self {
            data,
            peak_fits: Default::default(),
        }
    }

    fn fit_chunk_with(
        &self,
        chunk: PeakFitArgs<'_, '_>,
        config: FitConfig,
    ) -> (PeakShape, ModelFitResult) {
        let mut fits = Vec::new();

        let mut model = PeakShape::bigaussian(&chunk);
        let fit_result = model.fit_with(chunk.borrow(), config.clone());
        fits.push((model, fit_result));

        let (model, fit_result) = fits
            .into_iter()
            .min_by(|a, b| a.1.loss.total_cmp(&b.1.loss))
            .unwrap();
        (model, fit_result)
    }

    /// See [`PeakShapeFitter::fit_model`]
    pub fn fit_with(&mut self, config: FitConfig) {
        let partition_points = self.data.locate_extrema(None);
        let chunks = self.data.split_at(partition_points.as_slice());

        for chunk in chunks {
            let (model, fit_result) =
                self.fit_chunk_with(self.data.slice(chunk.clone()), config.clone());
            if fit_result.success {
                self.peak_fits.push(model);
            }
        }
    }

    /// See [`PeakShapeFitter::residuals`]
    pub fn residuals(&self) -> PeakFitArgs<'_, '_> {
        self.peak_fits.residuals(&self.data)
    }

    /// See [`PeakShapeFitter::predicted`]
    pub fn predicted(&self) -> PeakFitArgs<'_, '_> {
        let predicted = self
            .peak_fits
            .predict(&self.data.time)
            .into_iter()
            .map(|x| x as f32)
            .collect();
        let mut dup = self.data.borrow();
        dup.intensity = Cow::Owned(predicted);
        dup
    }

    /// See [`PeakShapeFitter::score`]
    pub fn score(&self) -> f64 {
        self.peak_fits.score(&self.data)
    }
}

#[cfg(test)]
mod test {
    use std::{
        fs,
        io::{self, prelude::*},
    };

    use log::debug;
    use mzpeaks::{feature::Feature, Time, MZ};

    use super::*;

    #[rstest::fixture]
    #[once]
    fn feature_table() -> Vec<Feature<MZ, Time>> {
        log::info!("Logging initialized");
        crate::text::load_feature_table("test/data/features_graph.txt").unwrap()
    }

    macro_rules! assert_is_close {
        ($t1:expr, $t2:expr, $tol:expr, $label:literal) => {
            assert!(
                ($t1 - $t2).abs() < $tol,
                "Observed {} {}, expected {}, difference {}",
                $label,
                $t1,
                $t2,
                $t1 - $t2,
            );
        };
        ($t1:expr, $t2:expr, $tol:expr, $label:literal, $obj:ident) => {
            assert!(
                ($t1 - $t2).abs() < $tol,
                "Observed {} {}, expected {}, difference {} from {:?}",
                $label,
                $t1,
                $t2,
                $t1 - $t2,
                $obj
            );
        };
    }

    #[rstest::rstest]
    #[test_log::test]
    fn test_fit_feature_14216(feature_table: &[Feature<MZ, Time>]) {
        let feature = &feature_table[14216];
        let args: PeakFitArgs = feature.into();

        let wmt = args.weighted_mean_time();
        assert_is_close!(wmt, 122.3535, 1e-3, "weighted mean time");

        let mut model = SkewedGaussianPeakShape::guess(&args);

        let expected_gradient = SkewedGaussianPeakShape {
            mu: 1.0877288990485208,
            sigma: 2.066092153296829,
            amplitude: 3421141.321363151,
            lambda: 0.846178224318954,
        };
        let gradient = model.gradient(&args);

        debug!("Initial:\n{model:?}");
        debug!("Gradient combo:\n{:?}", gradient);
        debug!("Gradient split:\n{:?}", model.gradient_split(&args));

        let _res = model.fit(args.borrow());
        let _score = model.score(&args);
        debug!("{model:?}\n{_res:?}\n{_score}\n");

        let expected = SkewedGaussianPeakShape {
            mu: 121.54820923262623,
            sigma: 0.14392304906433506,
            amplitude: 4768163.602247336,
            lambda: 0.055903399861434805,
        };

        assert_is_close!(expected.mu, model.mu, 1e-2, "mu");
        assert_is_close!(expected.sigma, model.sigma, 1e-3, "sigma");

        assert_is_close!(expected_gradient.mu, gradient.mu, 1e-2, "mu");
        assert_is_close!(expected_gradient.sigma, gradient.sigma, 1e-2, "sigma");
        // unstable
        // assert_is_close!(expected.lambda, model.lambda, 1e-3, "lambda");
        // assert_is_close!(expected.amplitude, model.amplitude, 100.0, "amplitude");
    }

    #[rstest::rstest]
    fn test_fit_feature_4490(feature_table: &[Feature<MZ, Time>]) {
        let feature = &feature_table[4490];
        let args = PeakFitArgs::from(feature);

        let expected_fits = MultiPeakShapeFit {
            fits: vec![
                PeakShape::BiGaussian(BiGaussianPeakShape {
                    mu: 125.41112342515179,
                    sigma_falling: 0.2130619068583823,
                    sigma_rising: 0.22703724439718917,
                    amplitude: 2535197.152987912,
                }),
                PeakShape::BiGaussian(BiGaussianPeakShape {
                    mu: 126.05807704271226,
                    sigma_falling: 0.9997523697939726,
                    sigma_rising: 1.1813146384558728,
                    amplitude: 267102.87981724285,
                }),
            ],
        };

        let mut fitter = SplittingPeakShapeFitter::new(args);
        fitter.fit_with(FitConfig::default().max_iter(10_000));
        debug!("Score: {}", fitter.score());
        debug!("Fits: {:?}", fitter.peak_fits);

        for (exp, obs) in expected_fits.iter().zip(fitter.peak_fits.iter()) {
            let expected_mu = dispatch_peak!(exp, model, model.mu);
            let observed_mu = dispatch_peak!(obs, model, model.mu);

            assert_is_close!(expected_mu, observed_mu, 1e-3, "mu");
        }
    }

    #[rstest::rstest]
    fn test_fit_feature_10979(feature_table: &[Feature<MZ, Time>]) {
        let feature = &feature_table[10979];
        let args: PeakFitArgs<'_, '_> = feature.into();

        let expected_split_point = SplittingPoint {
            first_maximum_height: 1562937.5,
            minimum_height: 130524.61,
            second_maximum_height: 524531.8,
            minimum_time: 127.1233653584,
        };

        let observed_split_point = args.locate_extrema(None).unwrap();

        assert_is_close!(
            expected_split_point.minimum_time,
            observed_split_point.minimum_time,
            1e-2,
            "minimum_time",
            observed_split_point
        );

        let mut fitter = SplittingPeakShapeFitter::new(args.borrow());
        fitter.fit_with(FitConfig::default().max_iter(10_000).smooth(3));
        debug!("Score: {}", fitter.score());
        debug!("Fits: {:?}", fitter.peak_fits);

        let expected_fits = MultiPeakShapeFit {
            fits: vec![
                PeakShape::BiGaussian(BiGaussianPeakShape {
                    mu: 125.46204779366751,
                    sigma_falling: 0.3419458944098784,
                    sigma_rising: 0.945137432308437,
                    amplitude: 1277701.0773096785,
                }),
                PeakShape::BiGaussian(BiGaussianPeakShape {
                    mu: 108.00971834292885,
                    sigma_falling: 0.09483862559250714,
                    sigma_rising: 0.32453016302695475,
                    amplitude: 487770.8422514298,
                }),
            ],
        };

        for (exp, obs) in expected_fits.iter().zip(fitter.peak_fits.iter()) {
            let expected_mu = dispatch_peak!(exp, model, model.mu);
            let observed_mu = dispatch_peak!(obs, model, model.mu);

            assert_is_close!(observed_mu, expected_mu, 1e-3, "mu");
        }
    }

    #[rstest::rstest]
    fn test_fit_args(feature_table: &[Feature<MZ, Time>]) {
        let features = feature_table;
        let feature = &features[160];
        let (_, y, z) = feature.as_view().into_inner();
        let args = PeakFitArgs::from((y, z));

        let wmt = args.weighted_mean_time();
        assert!(
            (wmt - 123.455).abs() < 1e-3,
            "Observed average weighted mean time {wmt}, expected 123.455"
        );

        let mut model = GaussianPeakShape::new(wmt, 1.0, 1.0);
        let _res = model.fit(args.borrow());
        let _score = model.score(&args);
        // eprint!("{model:?}\n{_res:?}\n{_score}\n");

        let mu = 123.44796615442881;
        let sigma = 0.1015352963957489;
        // let amplitude = 629639.6468112208;

        assert!(
            (model.mu - mu).abs() < 1e-3,
            "Model {0} found, expected {mu}, error = {1}",
            model.mu,
            model.mu - mu
        );
        assert!(
            (model.sigma - sigma).abs() < 1e-3,
            "Model {0} found, expected {sigma}, error = {1}",
            model.sigma,
            model.sigma - sigma
        );
        // Seems to be sensitive to the platform
        // assert!(
        //     (model.amplitude - amplitude).abs() < 1e-2,
        //     "Model {0} found, expected {amplitude}, error = {1}",
        //     model.amplitude,
        //     model.amplitude - amplitude
        // );
    }

    #[rstest::rstest]
    fn test_mixed_signal() {
        let time = vec![
            5., 5.05, 5.1, 5.15, 5.2, 5.25, 5.3, 5.35, 5.4, 5.45, 5.5, 5.55, 5.6, 5.65, 5.7, 5.75,
            5.8, 5.85, 5.9, 5.95, 6., 6.05, 6.1, 6.15, 6.2, 6.25, 6.3, 6.35, 6.4, 6.45, 6.5, 6.55,
            6.6, 6.65, 6.7, 6.75, 6.8, 6.85, 6.9, 6.95, 7., 7.05, 7.1, 7.15, 7.2, 7.25, 7.3, 7.35,
            7.4, 7.45, 7.5, 7.55, 7.6, 7.65, 7.7, 7.75, 7.8, 7.85, 7.9, 7.95, 8., 8.05, 8.1, 8.15,
            8.2, 8.25, 8.3, 8.35, 8.4, 8.45, 8.5, 8.55, 8.6, 8.65, 8.7, 8.75, 8.8, 8.85, 8.9, 8.95,
            9., 9.05, 9.1, 9.15, 9.2, 9.25, 9.3, 9.35, 9.4, 9.45, 9.5, 9.55, 9.6, 9.65, 9.7, 9.75,
            9.8, 9.85, 9.9, 9.95, 10., 10.05, 10.1, 10.15, 10.2, 10.25, 10.3, 10.35, 10.4, 10.45,
            10.5, 10.55, 10.6, 10.65, 10.7, 10.75, 10.8, 10.85, 10.9, 10.95, 11., 11.05, 11.1,
            11.15, 11.2, 11.25, 11.3, 11.35, 11.4, 11.45, 11.5, 11.55, 11.6, 11.65, 11.7, 11.75,
            11.8, 11.85, 11.9, 11.95,
        ];

        let intensity: Vec<f32> = vec![
            1.27420451e-10,
            6.17462536e-10,
            2.87663017e-09,
            1.28813560e-08,
            5.54347499e-08,
            2.29248641e-07,
            9.10983876e-07,
            3.47838473e-06,
            1.27613560e-05,
            4.49843188e-05,
            1.52358163e-04,
            4.95800813e-04,
            1.55018440e-03,
            4.65685268e-03,
            1.34410596e-02,
            3.72739646e-02,
            9.93134748e-02,
            2.54238248e-01,
            6.25321648e-01,
            1.47773227e+00,
            3.35519458e+00,
            7.31929673e+00,
            1.53408987e+01,
            3.08931557e+01,
            5.97728698e+01,
            1.11116052e+02,
            1.98463694e+02,
            3.40579046e+02,
            5.61550473e+02,
            8.89601967e+02,
            1.35407175e+03,
            1.98029962e+03,
            2.78272124e+03,
            3.75722702e+03,
            4.87459147e+03,
            6.07720155e+03,
            7.28110184e+03,
            8.38438277e+03,
            9.28130702e+03,
            9.87974971e+03,
            1.01181592e+04,
            9.97790016e+03,
            9.48776976e+03,
            8.71945410e+03,
            7.77507793e+03,
            6.76998890e+03,
            5.81486954e+03,
            5.00095885e+03,
            4.39083710e+03,
            4.01545364e+03,
            3.87648735e+03,
            3.95216934e+03,
            4.20449848e+03,
            4.58618917e+03,
            5.04639622e+03,
            5.53495197e+03,
            6.00532094e+03,
            6.41666569e+03,
            6.73537620e+03,
            6.93625265e+03,
            7.00335463e+03,
            6.93041211e+03,
            6.72066330e+03,
            6.38603273e+03,
            5.94566001e+03,
            5.42389927e+03,
            4.84799871e+03,
            4.24571927e+03,
            3.64315240e+03,
            3.06295366e+03,
            2.52313467e+03,
            2.03646669e+03,
            1.61046411e+03,
            1.24784786e+03,
            9.47346984e+02,
            7.04682299e+02,
            5.13587560e+02,
            3.66751988e+02,
            2.56606294e+02,
            1.75913423e+02,
            1.18159189e+02,
            7.77629758e+01,
            5.01435513e+01,
            3.16806551e+01,
            1.96114662e+01,
            1.18949556e+01,
            7.06890964e+00,
            4.11603334e+00,
            2.34823840e+00,
            1.31263002e+00,
            7.18917952e-01,
            3.85791959e-01,
            2.02844792e-01,
            1.04498823e-01,
            5.27467595e-02,
            2.60865722e-02,
            1.26408156e-02,
            6.00164112e-03,
            2.79191246e-03,
            1.27253700e-03,
            5.68297684e-04,
            2.48667027e-04,
            1.06609858e-04,
            4.47830199e-05,
            1.84317357e-05,
            7.43285977e-06,
            2.93685491e-06,
            1.13696185e-06,
            4.31266912e-07,
            1.60281439e-07,
            5.83656297e-08,
            2.08241826e-08,
            7.27973551e-09,
            2.49344667e-09,
            8.36799506e-10,
            2.75156384e-10,
            8.86491588e-11,
            2.79837874e-11,
            8.65516222e-12,
            2.62289424e-12,
            7.78795072e-13,
            2.26570027e-13,
            6.45830522e-14,
            1.80372998e-14,
            4.93584288e-15,
            1.32339040e-15,
            3.47657399e-16,
            8.94853257e-17,
            2.25677892e-17,
            5.57651734e-18,
            1.35012489e-18,
            3.20273997e-19,
            7.44399825e-20,
            1.69522634e-20,
            3.78256125e-21,
            8.26953508e-22,
            1.77138542e-22,
            3.71776457e-23,
            7.64517712e-24,
            1.54038778e-24,
        ];

        let args = PeakFitArgs::from((time, intensity));
        let split_point = args.locate_extrema(None).unwrap();
        assert_is_close!(split_point.minimum_time, 7.5, 1e-3, "minimum_height", split_point);

        let mut fitter = SplittingPeakShapeFitter::new(args);
        fitter.fit_with(FitConfig::default().max_iter(50_000));
        let score = fitter.score();
        assert!(score > 0.95, "Expected score {score} to be greater than 0.95");
    }
}
