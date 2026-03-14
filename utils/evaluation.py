import torch
import torch.nn as nn
import numpy as np


def evaluate_model(
    model,
    args,
    data,
    device,
    loss_fn=None,
    ponder_loss=None,
    ctm_gated_loss=None,
    use_ponder_loss=False,
    n_test_batches=-1,
):
    """
    Evaluate model on a dataset.

    Args:
        model: The model to evaluate
        args: Arguments namespace
        data: Dataset to evaluate on
        device: Device to run on
        loss_fn: Standard loss function (image_classification_loss)
        ponder_loss: PonderNet loss (optional)
        ctm_gated_loss: CTM-Gated loss (optional)
        use_ponder_loss: Whether to use ponder loss
        n_test_batches: Number of batches to evaluate (-1 for all)

    Returns:
        dict: Metrics including loss, accuracy, etc.
    """
    from utils.losses import image_classification_loss

    model.eval()
    all_targets_list = []
    all_predictions_list = []
    all_predictions_most_certain_list = []
    all_where_most_certain_list = []
    all_losses = []
    exit_steps_list = []
    certainty_at_exit_list = []
    certainty_per_step_list = []

    loader = torch.utils.data.DataLoader(
        data,
        batch_size=args.batch_size_test,
        shuffle=False,
        num_workers=1,
    )

    with torch.inference_mode():
        for inferi, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            all_targets_list.append(targets.detach().cpu().numpy())

            if args.model == "ctm":
                predictions, certainties, _ = model(inputs)
                if use_ponder_loss and ponder_loss is not None:
                    loss, p_t = ponder_loss(predictions, certainties.unsqueeze(1), targets)
                    where_to_eval = p_t.argmax(-1)
                    all_predictions_list.append(
                        predictions.argmax(1).detach().cpu().numpy()
                    )
                    all_predictions_most_certain_list.append(
                        predictions.argmax(1)[
                            torch.arange(predictions.size(0), device=predictions.device),
                            where_to_eval,
                        ].detach().cpu().numpy()
                    )
                    all_where_most_certain_list.append(where_to_eval.detach().cpu().numpy())
                else:
                    loss, where_most_certain = image_classification_loss(
                        predictions, certainties, targets, use_most_certain=True
                    )
                    all_predictions_list.append(
                        predictions.argmax(1).detach().cpu().numpy()
                    )
                    all_predictions_most_certain_list.append(
                        predictions.argmax(1)[
                            torch.arange(predictions.size(0), device=predictions.device),
                            where_most_certain,
                        ].detach().cpu().numpy()
                    )
                    all_where_most_certain_list.append(where_most_certain.detach().cpu().numpy())

            elif args.model == "ctm_gated":
                predictions, certainties, _, these_exit_steps, _ = model(inputs, use_early_exit=False)
                if ctm_gated_loss is not None:
                    loss, where_most_certain = ctm_gated_loss(
                        predictions, certainties, targets, these_exit_steps
                    )
                else:
                    loss, where_most_certain = image_classification_loss(
                        predictions, certainties, targets, use_most_certain=True
                    )
                all_predictions_list.append(predictions.argmax(1).detach().cpu().numpy())
                all_predictions_most_certain_list.append(
                    predictions.argmax(1)[
                        torch.arange(predictions.size(0), device=predictions.device),
                        where_most_certain,
                    ].detach().cpu().numpy()
                )
                all_where_most_certain_list.append(where_most_certain.detach().cpu().numpy())
                exit_steps_list.append(these_exit_steps.float().mean().item())
                certainty_at_exit = certainties[:, 1, :].gather(1, these_exit_steps.unsqueeze(1)).mean().item()
                certainty_at_exit_list.append(certainty_at_exit)
                certainty_per_step_list.append(certainties[:, 1, :].mean(dim=0).cpu().numpy())

            elif args.model == "lstm":
                predictions, certainties, _ = model(inputs)
                loss, where_most_certain = image_classification_loss(
                    predictions, certainties, targets, use_most_certain=True
                )
                all_predictions_list.append(predictions.argmax(1).detach().cpu().numpy())
                all_predictions_most_certain_list.append(
                    predictions.argmax(1)[
                        torch.arange(predictions.size(0), device=predictions.device),
                        where_most_certain,
                    ].detach().cpu().numpy()
                )
                all_where_most_certain_list.append(where_most_certain.detach().cpu().numpy())

            elif args.model == "ff":
                predictions = model(inputs)
                loss = nn.CrossEntropyLoss()(predictions, targets)
                all_predictions_list.append(predictions.argmax(1).detach().cpu().numpy())

            all_losses.append(loss.item())

            if n_test_batches != -1 and inferi >= n_test_batches - 1:
                break

    all_targets = np.concatenate(all_targets_list)
    all_predictions = np.concatenate(all_predictions_list)
    avg_loss = np.mean(all_losses)

    results = {"loss": avg_loss}

    if args.model in ["ctm", "lstm", "ctm_gated"]:
        current_accuracies = np.mean(
            all_predictions == all_targets[..., np.newaxis], axis=0
        )
        results["accuracies_per_tick"] = current_accuracies

        all_predictions_most_certain = np.concatenate(all_predictions_most_certain_list)
        results["accuracy_most_certain"] = (all_targets == all_predictions_most_certain).mean()

        if all_where_most_certain_list:
            all_where_most_certain = np.concatenate(all_where_most_certain_list)
            results["where_most_certain_mean"] = all_where_most_certain.mean()
            results["where_most_certain_median"] = np.median(all_where_most_certain)
            results["where_most_certain"] = all_where_most_certain

    else:
        results["accuracy"] = (all_targets == all_predictions).mean()

    if args.model == "ctm_gated":
        results["exit_steps"] = exit_steps_list
        results["certainty_at_exit"] = certainty_at_exit_list
        results["certainty_per_step"] = certainty_per_step_list

    return results


def compute_metrics_dict(results, args):
    """Convert evaluation results to wandb metrics dict."""
    log_dict = {}

    if "loss" in results:
        log_dict["Loss"] = results["loss"]

    if args.model in ["ctm", "lstm", "ctm_gated"]:
        log_dict["Accuracy (Most Certain)"] = results.get("accuracy_most_certain", 0)

        if "where_most_certain" in results:
            wc = results["where_most_certain"]
            log_dict["Most Certain Tick (Mean)"] = wc.mean()
            log_dict["Most Certain Tick (Median)"] = np.median(wc)
            log_dict["Most Certain Tick (Min)"] = wc.min()
            log_dict["Most Certain Tick (Max)"] = wc.max()
            log_dict["Most Certain Tick (Std)"] = wc.std()

        if "accuracies_per_tick" in results:
            for i, acc in enumerate(results["accuracies_per_tick"]):
                log_dict[f"Accuracy (Tick {i})"] = acc

        if args.model == "ctm_gated":
            if "exit_steps" in results and results["exit_steps"]:
                log_dict["Avg Exit Steps"] = np.mean(results["exit_steps"])
            if "certainty_at_exit" in results and results["certainty_at_exit"]:
                log_dict["Certainty at Exit"] = np.mean(results["certainty_at_exit"])
            if "certainty_per_step" in results and results["certainty_per_step"]:
                avg_certainty_per_step = np.mean(results["certainty_per_step"], axis=0)
                log_dict["Certainty (Step 5)"] = avg_certainty_per_step[4] if len(avg_certainty_per_step) > 4 else 0
                log_dict["Certainty (Step 10)"] = avg_certainty_per_step[9] if len(avg_certainty_per_step) > 9 else 0
                log_dict["Certainty (Final)"] = avg_certainty_per_step[-1]
    else:
        log_dict["Accuracy"] = results.get("accuracy", 0)

    return log_dict
